#![allow(unused)]
use ndarray::prelude::*;
use nih_plug::prelude::*;
use nih_plug_egui::{
    create_egui_editor,
    egui::{self, debug_text::print, Vec2},
    resizable_window::ResizableWindow,
    widgets, EguiState,
};
use realfft::{num_complex::Complex32, RealFftPlanner, RealToComplex};
use std::{
    collections::VecDeque,
    io::SeekFrom,
    sync::{atomic::AtomicI32, Arc, Mutex},
};
use tract_onnx::prelude::*;
mod filters;
use atomic_float::AtomicF32;
use egui::{Color32, FontId, RichText, Stroke};
use filters::*;
use std::sync::atomic::Ordering;

// This is a shortened version of the gain example with most comments removed, check out
// https://github.com/robbert-vdh/nih-plug/blob/master/plugins/examples/gain/src/lib.rs to get
// started

const MODEL_BYTES: &[u8] = include_bytes!("../model.onnx");

pub struct Chordetector {
    sample_rate: Arc<AtomicF32>,
    params: Arc<ChordetectorParams>,
    r2c_plan: Arc<dyn RealToComplex<f32>>,
    magnitude_buffer: Vec<f32>,
    complex_fft_buffer: Vec<Complex32>,
    downsampling_buffer: [f32; 3],
    downsampling_index: usize,
    fft_buffers: [[f32; 2048]; 8],
    fft_buffer_index: [usize; 8],
    mel_buffers: [Vec<f32>; 2],
    mel_buffer_index: [usize; 2],
    input_buffer: Array3<f32>,
    input_buffer_index: usize,
    result_buffer: Arc<Mutex<VecDeque<Result>>>,
    model: Arc<Option<SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>>>,
}

#[derive(Params)]
pub struct ChordetectorParams {
    /// The parameter's ID is used to identify the parameter in the wrappred plugin API. As long as
    /// these IDs remain constant, you can rename and reorder these fields as you wish. The
    /// parameters are exposed to the host in the same order they were defined. In this case, this
    /// gain parameter is stored as linear gain while the values are displayed in decibels.
    #[persist = "editor-state"]
    editor_state: Arc<EguiState>,
}

impl Default for Chordetector {
    fn default() -> Self {
        let mut planner = RealFftPlanner::new();
        let r2c_plan = planner.plan_fft_forward(2048);
        let mut real_fft_buffer = r2c_plan.make_input_vec();
        let mut complex_fft_buffer = r2c_plan.make_output_vec();

        real_fft_buffer[0..2048].copy_from_slice(&HAMMING);
        r2c_plan
            .process_with_scratch(&mut real_fft_buffer, &mut complex_fft_buffer, &mut [])
            .unwrap();
        Self {
            sample_rate: Arc::new(AtomicF32::new(1.0)),
            params: Arc::new(ChordetectorParams::default()),
            r2c_plan,
            complex_fft_buffer,
            model: Arc::new(Some(
                tract_onnx::onnx()
                    .model_for_read(&mut std::io::Cursor::new(MODEL_BYTES))
                    .unwrap()
                    .into_optimized()
                    .unwrap()
                    .into_runnable()
                    .unwrap(),
            )),
            downsampling_buffer: [0.0; 3],
            downsampling_index: 0,
            fft_buffer_index: [0, 256, 512, 768, 1024, 1280, 1536, 1792],
            fft_buffers: [[0.0; 2048]; 8],
            magnitude_buffer: vec![0.0; 1025],
            mel_buffer_index: [0, 8],
            mel_buffers: [vec![0.0; 2048], vec![0.0; 2048]],
            input_buffer: Array3::<f32>::zeros((4, 128, 16)),
            input_buffer_index: 0,
            result_buffer: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
}

impl Default for ChordetectorParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(300, 300),
        }
    }
}

impl Plugin for Chordetector {
    const NAME: &'static str = "Chordetector";
    const VENDOR: &'static str = "aizcutei";
    const URL: &'static str = env!("CARGO_PKG_HOMEPAGE");
    const EMAIL: &'static str = "aiz.cutei@gmail.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // The first audio IO layout is used as the default. The other layouts may be selected either
    // explicitly or automatically by the host or the user depending on the plugin API/backend.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),

        aux_input_ports: &[],
        aux_output_ports: &[],

        // Individual ports and the layout as a whole can be named here. By default these names
        // are generated as needed. This layout will be called 'Stereo', while a layout with
        // only one input and output channel would be called 'Mono'.
        names: PortNames::const_default(),
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    // If the plugin can send or receive SysEx messages, it can define a type to wrap around those
    // messages here. The type implements the `SysExMessage` trait, which allows conversion to and
    // from plain byte buffers.
    type SysExMessage = ();
    // More advanced plugins can use this to run expensive background tasks. See the field's
    // documentation for more information. `()` means that the plugin does not have any background
    // tasks.
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let sample_rate = self.sample_rate.clone();
        let egui_state = params.editor_state.clone();
        let result = self.result_buffer.clone();
        create_egui_editor(
            self.params.editor_state.clone(),
            (),
            |_, _| {},
            move |egui_ctx, setter, _state| {
                ResizableWindow::new("Chordetector")
                    .min_size(Vec2::new(100.0, 50.0))
                    .show(egui_ctx, egui_state.as_ref(), |ui| {
                        let fs = sample_rate.load(Ordering::Relaxed);
                        let result = result.lock().unwrap().clone();
                        if result.len() > 0 {
                            let last = result[result.len() - 1];

                            ui.label(
                                RichText::new(format!(
                                    "{}{}/{}",
                                    NOTE[last.chord / 62],
                                    CHORD[last.chord % 62],
                                    NOTE[last.note]
                                ))
                                .color(Color32::WHITE)
                                .font(FontId::proportional(40.0)),
                            );
                        }

                        if fs != 48000.0 {
                            ui.label(
                                RichText::new(
                                    "Sorry, this Plugin only works with a sample rate of 48000 Hz.",
                                )
                                .color(Color32::RED)
                                .font(FontId::proportional(12.0)),
                            );
                        }
                    });
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        // context.set_latency_samples(24000);
        self.sample_rate
            .store(buffer_config.sample_rate, Ordering::Release);
        if self.model.is_none() {
            self.model = Some(
                tract_onnx::onnx()
                    .model_for_read(&mut std::io::Cursor::new(MODEL_BYTES))
                    .unwrap()
                    .into_optimized()
                    .unwrap()
                    .into_runnable()
                    .unwrap(),
            )
            .into();
        }
        true
    }

    fn reset(&mut self) {
        // Reset buffers and envelopes here. This can be called from the audio thread and may not
        // allocate. You can remove this function if you do not need it.
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        if self.params.editor_state.is_open() {
            for channel_samples in buffer.iter_samples() {
                let sum = channel_samples
                    .into_iter()
                    .map(|sample| *sample)
                    .sum::<f32>();
                self.downsampling_buffer[self.downsampling_index] = sum;
                self.downsampling_index += 1;
                if self.downsampling_index == 3 {
                    self.downsampling_index = 0;
                    let average = self.downsampling_buffer.iter().sum::<f32>() / 6.0;

                    for i in 0..8 {
                        self.fft_buffers[i][self.fft_buffer_index[i]] =
                            average * HAMMING[self.fft_buffer_index[i]];
                        self.fft_buffer_index[i] += 1;
                        if self.fft_buffer_index[i] == 2048 {
                            self.fft_buffer_index[i] = 0;
                            let mut fft_buffer = self.fft_buffers[i].clone();
                            self.r2c_plan
                                .process(&mut fft_buffer, &mut self.complex_fft_buffer)
                                .unwrap();

                            for (j, val) in self
                                .complex_fft_buffer
                                .iter()
                                .take(1025)
                                .map(|c| c.norm())
                                .enumerate()
                            {
                                self.magnitude_buffer[j] = val;
                            }
                            let magnitude = ArrayView1::from(&self.magnitude_buffer[..]);
                            let mel_result = mel().dot(&magnitude) + 1.0;

                            for k in 0..2 {
                                self.mel_buffers[k][self.mel_buffer_index[k] * 128
                                    ..self.mel_buffer_index[k] * 128 + 128]
                                    .copy_from_slice(&mel_result.to_vec());
                                self.mel_buffer_index[k] += 1;
                                if self.mel_buffer_index[k] == 16 {
                                    self.mel_buffer_index[k] = 0;
                                    let mut input_single_slice = Array2::from_shape_vec(
                                        (16, 128),
                                        self.mel_buffers[k].clone(),
                                    )
                                    .unwrap()
                                    .reversed_axes()
                                    .ln();
                                    let std = input_single_slice.std(0.);
                                    if std == 0.0 {
                                        input_single_slice =
                                            Array2::from_elem((128, 16), 0.000001f32);
                                    } else {
                                        let mean = input_single_slice.mean().unwrap();
                                        input_single_slice =
                                            (input_single_slice - mean) * std.recip();
                                    }
                                    self.input_buffer
                                        .slice_mut(s![self.input_buffer_index, .., ..])
                                        .assign(&input_single_slice);
                                    self.input_buffer_index += 1;
                                    let input_tensor: Tensor = self.input_buffer.clone().into();
                                    if self.input_buffer_index == 4 {
                                        self.input_buffer_index = 0;
                                        let result = self
                                            .model
                                            .as_ref()
                                            .clone()
                                            .unwrap()
                                            .run(tvec!(input_tensor.into()))
                                            .unwrap();

                                        for l in 0..4 {
                                            let (_, note) = result[0]
                                                .to_array_view::<f32>()
                                                .unwrap()
                                                .to_slice()
                                                .unwrap()
                                                .iter()
                                                .clone()
                                                .skip(l * 12)
                                                .take(12)
                                                .zip(0usize..)
                                                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                                                .unwrap();
                                            let (_, chord) = result[1]
                                                .to_array_view::<f32>()
                                                .unwrap()
                                                .to_slice()
                                                .unwrap()
                                                .iter()
                                                .clone()
                                                .skip(l * 62 * 12)
                                                .take(62 * 12)
                                                .zip(0usize..)
                                                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                                                .unwrap();
                                            self.result_buffer
                                                .lock()
                                                .unwrap()
                                                .push_back(Result { note, chord });
                                            if self.result_buffer.lock().unwrap().len() >= 10 {
                                                self.result_buffer.lock().unwrap().pop_front();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for Chordetector {
    const CLAP_ID: &'static str = "com.aizcutei.chordetector";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Detect chord from audio");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    // Don't forget to change these features
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Stereo];
}

impl Vst3Plugin for Chordetector {
    const VST3_CLASS_ID: [u8; 16] = *b"AizcChordetector";

    // And also don't forget to change these categories
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Dynamics];
}

nih_export_clap!(Chordetector);
nih_export_vst3!(Chordetector);
