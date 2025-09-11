// use crate::gstar_file::Sequence;

use crate::traits::Backend;

// Main function to simulate file parsing
// mod gstar_file;
mod sequence;
mod traits;
mod gs_backend;
mod types;
mod gs_sequence;
mod util;

fn main() {
    // Example of using a path, adjust the path as needed.
    let path_to_file = "";
    // let result  = sequence::Sequence::form_file(path_to_file);  
    let result = gs_backend::GstarSequence::load(path_to_file).unwrap();
    // result.duration();
    
    // let t = f64::NEG_INFINITY;
    let tt = 0.0;
    // let s = result.events(types::EventType::Gradient(types::GradientChannel::X), t, 7000.0, 10000);
    // let s = result.encounter(t, types::EventType::Gradient(types::GradientChannel::X));
    let s = result.integrate(&[tt, 0.007]);
    let en = result.encounter(tt, types::EventType::RfPulse);
    println!("{:?}",en );
    println!("finish" );
    let mut t = 0.0;
    let seq:sequence::Sequence = sequence::Sequence(Box::new(result));
    let dur = seq.duration();
    while let Some((pulse_start, pulse_end)) = seq.encounter(t,
         types::EventType::RfPulse) {
        let types::Moment { pulse, .. } = seq.integrate_one(pulse_start, pulse_end);
        let x = seq.events(types::EventType::Adc, pulse_start, pulse_end, 80);

        println!(
            "[{pulse_start}: {}ms]: {pulse:?}  abd {x:?} and duration is {dur}",
            (pulse_end - pulse_start) * 1e3
        );
        t = pulse_end;
    }
}
