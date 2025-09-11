
use ezpc::*;
use serde::{Deserialize, Serialize};
use serde_json::{Result as JsonResult, Value};
use std::collections::HashMap;
use std::fs::File;

use std::io::{self, BufRead, BufReader};
use std::path::Path;

// use crate::sequence::Section;
mod helpers;

// use mod helpers
#[derive(Serialize, Deserialize, Debug)]
pub struct Sequence {
    pub blocks: Vec<Block>,
}
pub enum Section {
    Definitions(Vec<(String, String)>),
    Blocks(Vec<Block>),
    Rfs(Vec<RfEvent>),
    // Gradients(Vec<Gradient>),
    // Traps(Vec<Trap>),
    Adcs(Vec<AdcHeader>),
    // Delays(Vec<Delay>),
    // Extensions(Extensions),
    // Shapes(Vec<Shape>),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Block {
    pub tstart: u32,
    pub duration: u32,
    pub has_trigger: bool,
    pub trigger_option: Option<String>,
    pub trigger_duration: Option<u32>,
    pub rf_pulse_id: i8,
    pub rf: Option<RfEvent>,
    
    pub gradx_v: Vec<f64>,
    pub gradx_t: Vec<u32>,
    pub gradx_tstart: Option<u32>,

    pub grady_v: Vec<f64>,
    pub grady_t: Vec<u32>,
    pub grady_tstart: Option<u32>,

    pub gradz_v: Vec<f64>,
    pub gradz_t: Vec<u32>,
    pub gradz_tstart: Option<u32>,

    pub has_adc: bool,
    pub adc_tstart: Option<u32>,
    pub adc_header: Option<AdcHeader>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RfEvent {
    pub rf_type: String,
    pub rf_am: Vec<f64>,
    pub rf_phs: f64,
    pub rf_fm: Vec<f64>,
    // pub rf_asym: f64,
    pub rf_dur: Option<u32>,
    pub rf_tstart: u32,
    pub rf_frq: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Gradient {
    pub grad_v: Vec<f64>,
    pub grad_t: Vec<u32>,
    pub grad_tstart: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AdcEvent {
    pub adc_header: AdcHeader,
    
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AdcHeader {
    pub sample_time_us: f64,
    pub number_of_samples: u32,
    pub position: Vec<f64>,
    pub slice_dir: Vec<f64>,
    pub read_dir: Vec<f64>,
    pub phase_dir: Vec<f64>,
    pub center_sample: u32,
    // Add additional fields from ADC header as needed
}

#[derive(Debug)]
pub struct Version {
    pub identifier: String,
}


// pub fn parse_file(source: &str) {
//     let block: &str = ;
//     file().parse_all(source)

// }



// pub fn file() -> Parser<impl Parse<Output=Section>>{

//     // rfs().map(Section::Rfs);
//     adcs().map(Section::Adcs)
// }



// pub fn rfs() -> Parser<impl Parse<Output = Vec<RfEvent>>> {
//     let i = || helpers::ws() + helpers::int();
//     let f = || helpers::ws() + helpers::float();
//     let rf = (helpers::ws().opt()
//      + helpers::int() + f() + i() + i() + i() + i() + f() + f()).map(
//         |(((((rf_type, rf_am), rf_fm),
//          rf_tstart), rf_dur), rf_frq)| RfEvent {
//             rf_type,
//             rf_am,
//             rf_fm,
            
//             rf_tstart,
//             // rf_asym,
//             rf_dur,
            
//             rf_frq,
//             rf_phs,
//         },
//     );
//     tag_nl("[RF]") + (rf + nl()).repeat(0..)
// }

// pub fn adcs() -> Parser<impl Parse<Output = Vec<AdcHeader>>> {
//     // let adc = (ws().opt() + float
//     // send dummy value
//     let v : Vec<AdcHeader> = Vec::new();
    
//     helpers::tag_nl("[ADC]") + (v + helpers::nl()).repeat(0..)
// }

// Main function to simulate file parsing
// pub fn test(file_path:&str) -> JsonResult<()> {
//     let path = Path::new(file_path);
//     let file = File::open(path).expect("Unable to open file");
//     let reader = BufReader::new(file);

//     let mut sequence = Sequence { blocks: Vec::new() };

//     for line in reader.lines() {
//         let line: String = line.expect("Unable to read line");
//         let block:Block = serde_json::from_str(&line).expect("Unable to parse block");
//         let _hashmap: HashMap<String, Value> = serde_json::from_str(&line)?;
//         if block.rf_pulse_id == 1 {
//              // put rf related into into rf_event struct optional field
//         }
//         match serde_json::from_str::<Block>(&line) {
//             Ok(block) => sequence.blocks.push(block),
//             Err(e) => eprintln!("Error parsing block: {:?}", e),
//         }
//     }

//     println!("Parsed sequence with {} blocks", sequence.blocks.len());
//     Ok(())
// }