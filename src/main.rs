use chordetector::Chordetector;
use nih_plug::prelude::*;
fn main() {
    nih_export_standalone::<Chordetector>();
}
