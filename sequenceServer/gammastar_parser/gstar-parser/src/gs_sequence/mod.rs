#![allow(dead_code)]
#![allow(unused_imports)]
use std::{ fs::File,  path::Path, sync::Arc};
use std::io::{self, BufRead, BufReader};
// use serde::de::value::Error;
use serde::{Deserialize, Serialize, Deserializer};
use std::collections::BTreeMap;
use self::from_raw::{create_gradient, grad_type};
// use crate::{error, gstar_file};
mod from_raw;
pub struct  Sequence {
    pub time_raster: f64,
    pub name: Option<String>,
    pub fov: Option<(f64, f64, f64)>,
    pub blocks: Vec<NestedBlock>,
}


impl Sequence {

    // pub fn from_parsed_file(sections:Vec<Section>) {
    //     let tmp:Sequence = from_raw::from_raw(sections)?;
    //     tmp.validate()?;
    //     Ok(tmp)
    // }

    // pub fn from_parsed_file(sections:Vec<Section>) -> Result<Self, error::Error>{
    //     let tmp:Sequence = from_raw::from_raw(sections)?;
    //     tmp.validate()?;
    //     Ok(tmp)
    // }
    
    // pub fn from_source(source: &str)  {
    //     let sections = gstar_file::parse_file(source);
    //     Self::from_parsed_file(sections)
    // }

    pub fn form_file<P: AsRef<Path>>(path: P)-> Sequence {

        // let path = Path::new(path);
        let file = File::open(path).expect("Unable to open file");
        let reader = BufReader::new(file);
// 
        let mut sequence = Sequence { blocks: Vec::new(),
        time_raster: 10.0 * 1e-6,
        name:Some("gamma_star_generic".to_string()),
        fov: Some((0.256, 0.256, 0.005)),};

        for line in reader.lines() {
            let line: String = line.expect("Unable to read line");

            // let block:Block = serde_json::from_str(&line).expect("Unable to parse block");
            
        
            match serde_json::from_str::<RawBlock>(&line) {

                Ok(rawblock) =>
                {    
                    let block = rawblock.into();
                    let nested_block = NestedBlock::from_block(&block);
                    sequence.blocks.push(nested_block)}
                Err(e) => eprintln!("Error parsing block: {:?}", e),
            }
    }
    return sequence;
    }

}
#[derive(Serialize, Deserialize)]
pub struct Block {
    
    pub tstart: f64,
    pub duration: f64,
    pub has_trigger: bool,
    pub trigger_option: Option<String>,
    pub trigger_duration: Option<u32>,


    pub rf_pulse_id: i16,
    // pub rf: Option<Rf>,
    pub rf_am: Option<Vec<f64>>,
    pub rf_fm: Option<Vec<f64>>,
    pub rf_phs: Option<f64>,
    pub rf_dur: Option<u32>,
    pub rf_frq: Option<f64>,
    pub rf_t: Option<Vec<f64>>,
    pub rf_tstart: Option<f64>,
    pub rf_asym: Option<f64>,
    pub rf_type: Option<String>,

    //pub rf: Option<RfEvent>,
    
    pub gradx_v: Vec<f64>,
    pub gradx_t: Vec<f64>,
    pub gradx_tstart: Option<f64>,

    pub grady_v: Vec<f64>,
    pub grady_t: Vec<f64>,
    pub grady_tstart: Option<f64>,

    pub gradz_v: Vec<f64>,
    pub gradz_t: Vec<f64>,
    pub gradz_tstart: Option<f64>,

    pub adc_phs: Option<f64>,
    pub has_adc: bool,
    pub adc_tstart: Option<f64>,
    pub adc_header: Option<AdcRaw>,
}

#[derive(Deserialize)]
struct RawBlock {
    #[serde(default)]
    tstart: f64,
    duration: f64,
    has_trig: bool,
    trig_mode: Option<String>,
    trig_dur: Option<u32>,
    rf_id: i16,
    rf_v: Option<BTreeMap<usize, RfChannel>>,  // ✅ Now `rf_v` is a map `{}` instead of a list `[]`
    rf_phs: Option<f64>,
    rf_dur: Option<u32>,
    rf_frq: Option<f64>,
    rf_t: Option<BTreeMap<usize, f64>>,
    rf_tstart: Option<f64>,
    rf_asym: Option<f64>,
    rf_type: Option<String>,
    gradx_v: BTreeMap<usize, f64>,
    gradx_t: BTreeMap<usize, f64>,
    gradx_tstart: Option<f64>,
    grady_v: BTreeMap<usize, f64>,
    grady_t: BTreeMap<usize, f64>,
    grady_tstart: Option<f64>,
    gradz_v: BTreeMap<usize, f64>,
    gradz_t: BTreeMap<usize, f64>,
    gradz_tstart: Option<f64>,
    adc_phs: Option<f64>,
    has_adc: bool,
    adc_tstart: Option<f64>,
    adc_header: Option<RawAdcHeader>
}

#[derive(Deserialize)]
struct RfChannel {
    #[serde(default, deserialize_with = "map_to_sorted_vec")]
    am: Vec<f64>,  // ✅ Ensures `am: {"1": 0.1, "2": 0.2} → [0.1, 0.2]`
    
    #[serde(default, deserialize_with = "map_to_sorted_vec")]
    fm: Vec<f64>,  // ✅ Ensures `fm: {"1": 1.0, "2": 2.0} → [1.0, 2.0]`
}

impl  Block {

    fn new()-> Self {
        Self{
            rf_tstart: Some(0.0),
            rf_dur: Some(0),
            tstart: 0.0,
            duration: 0.0,
            has_trigger: false,
            trigger_option: None,
            trigger_duration: None,
            rf_pulse_id: 0,
            rf_t: Some(Vec::new()),
            // rf: None,
            rf_am: None,
            rf_fm: None,
            rf_phs: None,
            rf_frq: None,
            rf_asym: None,
            rf_type: None,
            // rf_delay: 0.0,
            gradx_v: Vec::new(),
            gradx_t: Vec::new(),
            gradx_tstart: None,
            grady_v: Vec::new(),
            grady_t: Vec::new(),
            grady_tstart: None,
            gradz_v: Vec::new(),
            gradz_t: Vec::new(),
            gradz_tstart: None,

            adc_phs: Some(0.0),
            adc_tstart: Some(0.0),
            has_adc: false,
            adc_header: None,

        }
        
    }
    
}

impl From<RawBlock> for Block {
    fn from(raw_block: RawBlock) -> Self {
        let rf_v = raw_block.rf_v.unwrap_or_default();
        Block {
            tstart: raw_block.tstart,
            duration: raw_block.duration,
            has_trigger: raw_block.has_trig,
            trigger_option: raw_block.trig_mode,
            trigger_duration: raw_block.trig_dur,
            rf_pulse_id: raw_block.rf_id,
            
            rf_am: Some(rf_v.values().flat_map(|v| v.am.clone()).collect()),
            rf_fm: Some(rf_v.values().flat_map(|v| v.fm.clone()).collect()),
            rf_phs: raw_block.rf_phs,
            rf_dur: raw_block.rf_dur,
            rf_frq: raw_block.rf_frq,
            rf_t: raw_block.rf_t.map(|hm| hm.into_iter().map(|(_, v)| v).collect()),
            rf_tstart: raw_block.rf_tstart,
            rf_asym: raw_block.rf_asym,
            rf_type: raw_block.rf_type,
            gradx_v: raw_block.gradx_v.into_iter().map(|(_, v)| v).collect(),
            gradx_t: raw_block.gradx_t.into_iter().map(|(_, v)| v).collect(),
            gradx_tstart: raw_block.gradx_tstart,
            grady_v: raw_block.grady_v.into_iter().map(|(_, v)| v).collect(),
            grady_t: raw_block.grady_t.into_iter().map(|(_, v)| v).collect(),
            grady_tstart: raw_block.grady_tstart,
            gradz_v: raw_block.gradz_v.into_iter().map(|(_, v)| v).collect(),
            gradz_t: raw_block.gradz_t.into_iter().map(|(_, v)| v).collect(),
            gradz_tstart: raw_block.gradz_tstart,
            adc_phs: raw_block.adc_phs,
            has_adc: raw_block.has_adc,
            adc_tstart: raw_block.adc_tstart,
            adc_header: raw_block.adc_header.map(|raw| raw.into()),
        }
    }
}


pub struct NestedBlock {
    pub tstart: f64,
    pub duration: f64,
    pub rf: Option<Arc<Rf>>,
    pub gx: Option<Arc<Gradient>>,
    pub gy: Option<Arc<Gradient>>,
    pub gz: Option<Arc<Gradient>>,
    pub adc: Option<Arc<Adc>>,
}

impl NestedBlock {
    pub fn from_block(block: &Block) -> NestedBlock {
        let rf = if block.rf_pulse_id > 0 {
            Some(Arc::new(Rf {
                
                amp: block.rf_amp().unwrap_or_default() * 42.58, //max amp in Hz
                phase: block.rf_phase().unwrap_or_default(), //phase in rad
                delay: block.rf_delay().unwrap_or_default() *1e-6 , //delay in s
                freq: block.rf_frq.unwrap_or_default(), //freq in Hz
                amp_shape: block.rf_amp_shape().unwrap(), //shape of the amplitude scale 0-1
                rf_t: block.rf_t.as_ref().unwrap().clone().into_iter().
                map(|x| x*1e-6).collect(),
                // rf_t: block.rf_interp_amp_shape().
                // unwrap().1.into_iter().
                // map(|x| x*1e-6).collect(),
                phase_shape: block.rf_phase_shape().expect("error getting rf_phase_shape"),
                rf_dur: block.rf_duration().unwrap_or_default()*1e-6,
            }))
        } else {
            None
        };
    
        let gx = if block.gradx_tstart.is_some() {
            create_gradient(&block.gradx_v, &block.gradx_t,
                 block.gradx_tstart.unwrap_or_default())
        } else {
            None
        };
        let gy = if block.grady_tstart.is_some() {
            create_gradient(&block.grady_v, &block.grady_t,
                block.grady_tstart.unwrap_or_default())
        } else {
            None
        };
    
        let gz = if block.gradz_tstart.is_some() {
            create_gradient(&block.gradz_v, &block.gradz_t,
                block.gradz_tstart.unwrap_or_default())
        } else {
            None
        };
    
        let adc = if block.has_adc {
            Some(Arc::new(Adc {
                num: block.adc_header.as_ref().unwrap().number_of_samples,
                dwell: block.adc_header.as_ref().unwrap().sample_time_us * 1e-6,
                delay: block.adc_delay().unwrap_or_default() * 1e-6,
                freq: block.adc_header.as_ref().unwrap().adc_freq().unwrap_or_default(),
                phase: block.adc_phs.unwrap_or_default(),
                adc_dur: block.adc_duration().unwrap_or_default() * 1e-6,

            }))
        } else {
            None
        };
    
        NestedBlock {
            // Other fields...
            tstart: block.tstart *1e-6,
            duration: block.duration *1e-6,
            rf,
            gx,
            gy,
            gz,
            adc,
        }
    }
    
}

// #[derive(Serialize, Deserialize)]
pub struct Rf {
    /// Unit: `[Hz]`
    pub amp: f64,
    /// Unit: `[rad]`
    pub phase: f64,
    /// Unit: `[s]`
    pub delay: f64,
    /// Unit: `[Hz]`
    pub freq: f64,
    // Shapes
    pub amp_shape: Arc<Shape>,
    pub phase_shape: Arc<Shape>,
    pub rf_t: Vec<f64>,
    pub rf_dur: f64,
}

#[derive(Serialize, Deserialize)]
pub struct Adc {
    pub num: u32,
    /// Unit: `[us]`
    pub dwell: f64,
    /// Unit: `[us]`
    pub delay: f64,
    /// Unit: `[Hz]`
    pub freq: f64,
    /// Unit: `[rad]`
    pub phase: f64,
    
    pub adc_dur: f64,

}

#[derive(Serialize, Deserialize, Debug)]
pub struct AdcRaw {
    pub sample_time_us: f64,
    pub number_of_samples: u32,
    pub position: Vec<f64>,
    pub slice_dir: Vec<f64>,
    pub read_dir: Vec<f64>,
    pub phase_dir: Vec<f64>,
    pub center_sample: u32,
    pub idx_kspace_encode_step_1: Option<u32>,
    pub flex_encoding_encodingLimits_kspace_encoding_step_2: Option<Vec<u32>>,
    pub flex_encoding_encodingLimits_kspace_encoding_step_1: Option<Vec<u32>>,
    pub ACQ_FIRST_IN_REPETITION: Option<bool>,
    pub ACQ_FIRST_IN_ENCODE_STEP1: Option<bool>,
    pub ACQ_FIRST_IN_SLICE: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RawAdcHeader {
    pub sample_time_us: f64,
    pub number_of_samples: u32,
    pub position: BTreeMap<usize, f64>,
    pub slice_dir: BTreeMap<usize, f64>,
    pub read_dir: BTreeMap<usize, f64>,
    pub phase_dir: BTreeMap<usize, f64>,
    pub center_sample: u32,
    pub idx_kspace_encode_step_1: Option<u32>,
    pub flex_encoding_encodingLimits_kspace_encoding_step_2: Option<BTreeMap<usize, u32>>,
    pub flex_encoding_encodingLimits_kspace_encoding_step_1: Option<BTreeMap<usize, u32>>,
    pub ACQ_FIRST_IN_REPETITION: Option<bool>,
    pub ACQ_FIRST_IN_ENCODE_STEP1: Option<bool>,
    pub ACQ_FIRST_IN_SLICE: Option<bool>,
}

impl From<RawAdcHeader> for AdcRaw {
    fn from(raw: RawAdcHeader) -> Self {
        AdcRaw {
            sample_time_us: raw.sample_time_us,
            number_of_samples: raw.number_of_samples,
            position: convert_BTreeMap_to_vec(&raw.position),
            slice_dir: convert_BTreeMap_to_vec(&raw.slice_dir),
            read_dir: convert_BTreeMap_to_vec(&raw.read_dir),
            phase_dir: convert_BTreeMap_to_vec(&raw.phase_dir),
            center_sample: raw.center_sample,
            idx_kspace_encode_step_1: raw.idx_kspace_encode_step_1,
            flex_encoding_encodingLimits_kspace_encoding_step_2: raw
                .flex_encoding_encodingLimits_kspace_encoding_step_2
                .map(|hm| convert_BTreeMap_to_vec(&hm)),
            flex_encoding_encodingLimits_kspace_encoding_step_1: raw
                .flex_encoding_encodingLimits_kspace_encoding_step_1
                .map(|hm| convert_BTreeMap_to_vec(&hm)),
            ACQ_FIRST_IN_REPETITION: raw.ACQ_FIRST_IN_REPETITION,
            ACQ_FIRST_IN_ENCODE_STEP1: raw.ACQ_FIRST_IN_ENCODE_STEP1,
            ACQ_FIRST_IN_SLICE: raw.ACQ_FIRST_IN_SLICE,
        }
    }
}

fn convert_BTreeMap_to_vec<T>(map: &BTreeMap<usize, T>) -> Vec<T>
where
    T: Default + Clone,
{
    let mut vec = vec![T::default(); map.len()];
    for (key, value) in map {
        // Since `key` is already a `usize`, no need to parse
        if *key > 0 && *key <= vec.len() {
            vec[key - 1] = value.clone();
        }
    }
    vec
}

#[derive(Serialize, Deserialize)]
pub struct GradientRaw {
    pub grad_v: Vec<f64>,
    pub grad_t: Vec<f64>,
    pub grad_tstart: f64,
}

pub enum Gradient {
    Free {
        /// Unit: `[Hz/m]`
        amp: f64,
        /// Unit: `[s]`
        delay: f64,
        // Shapes
        shape: Arc<Shape>,
        times: Vec<f64>,
        
        dur: f64,
    },
    Trap {
        /// Unit: `[Hz/m]`
        amp: f64,
        /// Unit: `[s]`
        rise: f64,
        /// Unit: `[s]`
        flat: f64,
        /// Unit: `[s]`
        fall: f64,
        /// Unit: `[s]`
        delay: f64,

        dur: f64,
    },
}
pub struct Shape(pub Vec<f64>);


impl Shape {
    // Converts RF amplitude from microtesla to Hertz
    pub fn normalize(&self, max_amp:f64) -> Shape {
        // Gyromagnetic ratio for protons in Hz/μT
        // let gamma_hz_per_ut = 42.58;
        
        // Convert each amplitude from μT to Hz
        let converted_amplitudes = self.0.iter().map(|&amp| amp/max_amp).collect();
        
        Shape(converted_amplitudes)
    }
}

pub enum Section {
    Definitions(Vec<(String, String)>),
    Blocks(Vec<Block>),
    Rfs(Vec<Rf>),
    // Gradients(Vec<Gradient>),
    // Traps(Vec<Trap>),
    Adcs(Vec<Adc>),
    // Delays(Vec<Delay>),
    // Extensions(Extensions),
    // Shapes(Vec<Shape>),
}


impl GradientRaw {

    pub fn grad_type(&self) -> &str {
        // Assume self.grad_v is of type Option<Vec<f64>>
        if !self.grad_v.is_empty() {

            // Check if the vector is exactly of length 4
            if self.grad_v.len() == 4 {
                "trap"
            } else {
                "free"
            }
        } else {
            // grad_v is None
            "none"
        }
    }
    
    
}

fn map_to_sorted_vec<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: serde_json::Value = Deserialize::deserialize(deserializer)?;
    match value {
        serde_json::Value::Array(vec) => {
            let result: Result<Vec<f64>, _> = vec.into_iter().map(serde_json::from_value).collect();
            result.map_err(serde::de::Error::custom)
        }
        serde_json::Value::Object(map) => {
            let mut sorted_values = vec![0.0; map.len()];
            for (key, val) in map {
                if let Ok(index) = key.parse::<usize>() {
                    if let serde_json::Value::Number(num) = val {
                        if let Some(f) = num.as_f64() {
                            if index > 0 && index <= sorted_values.len() {
                                sorted_values[index - 1] = f;
                            }
                        }
                    }
                }
            }
            Ok(sorted_values)
        }
        _ => Err(serde::de::Error::custom("Expected array or map")),
    }
}
