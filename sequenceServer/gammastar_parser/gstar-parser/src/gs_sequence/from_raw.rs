#[allow(unused_imports)]
use std::{
    collections::{hash_map::Entry, HashMap},
    hash::Hash,
};
use crate::util::{linspace, linear_interp};
// #[allow(unused_imports)]
// use crate::error::ConversionError;

use super::*;
// pub fn from_raw(mut sections:Vec<Section>) -> Result<Sequence, ConversionError> {
    
// }


impl Block {
   pub fn rf_delay(&self) -> Option<f64> {
    //TODO: check if this is correct
        Some(self.rf_tstart.unwrap_or(0.0))
    }

    pub fn rf_amp(&self) -> Option<f64> {
        
        let amp = self.rf_am.as_ref().and_then(|amps|
            amps.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        ).copied();
        Some(amp.unwrap_or(0.0))
        
    }

    pub fn rf_phase (&self) -> Option<f64> {
        Some(self.rf_phs.unwrap_or(0.0))
    }

    pub fn rf_frq (&self) -> Option<f64> {
        Some(self.rf_frq.unwrap_or(0.0))
    }

    pub fn rf_amp_shape(&self) -> Option<Arc<Shape>> {
        self.rf_am.as_ref().map(|amplitudes| {
            Arc::new(Shape(amplitudes.clone()).normalize(self.rf_amp().unwrap_or(1.0) as f64))
        }) 
    
    }

    pub fn rf_interp_amp_shape(&self) -> Option<(Arc<Shape>, Vec<f64>)> {

        let amp_shape= self.rf_am.as_ref()?;
        let amp_times = self.rf_t.as_ref()?;
        let max_amp = self.rf_amp().unwrap_or(0.0);
        let times = linspace(0.0,
             *amp_times.last().unwrap()+ 5.0,
         10.0);
        let amp_shape_interp = linear_interp(amp_times,
             amp_shape,
              &times,
              &max_amp);
        
        Some((Arc::new(Shape(amp_shape_interp)), times))
    }


    pub fn rf_phase_shape(&self) -> Option<Arc<Shape>> {
        //return a shape of vector of length equal to rf_am but zeros
        Some(Arc::new(Shape(self.rf_amp_shape().as_ref()
        .map(|amps| vec![0.0; amps.0.len()]).unwrap_or_else(|| vec![]))))
    }

    pub fn rf_duration(&self) -> Option<f64> {
        // Some(self.rf_t.as_ref().map(|times| *times.last().unwrap_or(&0.0)).unwrap_or(0.0))
        Some(self.rf_dur.unwrap_or(0) as f64 + self.rf_tstart.unwrap_or(0.0) )
    }


    pub fn adc_delay(&self) -> Option<f64> {
        Some(self.adc_tstart.unwrap_or(0.0))
    }

    
    pub fn adc_phase(&self) -> Option<f64> {
        Some(self.adc_phs.unwrap_or(0.0))
    }

    pub fn adc_duration(&self) -> Option<f64> {
        Some(self.adc_header.as_ref().map(|adc| adc.sample_time_us * adc.number_of_samples as f64).unwrap_or(0.0))
    }



}
use std::{ops::Deref, sync::Arc};
impl Gradient {
    // pub fn grad_type(&self) -> f64 {
    //     self.grad_tstart - block.tstart
    // }
    
    }

pub fn grad_type(grad_v: &Vec<f64>) -> &str {
    // Assume self.grad_v is of type Option<Vec<f64>>
    if !grad_v.is_empty() {

        // Check if the vector is exactly of length 4
        //TODO just a test to turn every grad into free grad
        if grad_v.len() == 1 {
            "trap"
        } else {
            "free"
        }
    } else {
        // grad_v is None
        "none"
    }


}

pub fn grad_interp_amp_shape(amp_shape:&Vec<f64>, amp_times:&Vec<f64>) -> Option<(Arc<Shape>, Vec<f64>)> {
    

    let times = linspace(0.0,
         *amp_times.last().unwrap(),
          10.0);
    let amp_shape_interp = linear_interp(amp_times, amp_shape, &times, &(1.0/(42.58*1e3)));
    
    Some((Arc::new(Shape(amp_shape_interp)), times.iter().map(|x| x*1e-6).collect()))
}


pub fn grad_duration(grad_t: &Vec<f64>, delay:f64) -> f64 {
    grad_t.last().unwrap_or(&0.0) + delay
}



pub fn create_gradient(grad_v: &Vec<f64>, times: &Vec<f64>, start_time: f64) -> Option<Arc<Gradient>> {
    let grad_type = grad_type(grad_v);
    match grad_type {
        "trap" => {

                Some(Arc::new(Gradient::Trap {
                    amp: grad_v[1]* 42.58*1e3,
                    rise: (times[1] -times[0])* 1e-6 ,
                    flat:( times[2]-times[1])* 1e-6,
                    fall: (times[3]-times[2])* 1e-6,
                    delay: start_time * 1e-6,
                    dur: grad_duration(times, start_time) * 1e-6,
                }))
            } 
        ,
        "free" => Some(Arc::new(Gradient::Free {
                    amp: grad_v[1] * 42.58*1e3, //convert to Hz/m
                    delay: start_time * 1e-6, 
                    shape: grad_interp_amp_shape(grad_v, times).unwrap().0,
                    times: grad_interp_amp_shape(grad_v, times).unwrap().1,
                    dur: grad_duration(times, start_time) * 1e-6,

                })),
        "none" => Some(Arc::new(Gradient::Free {
            amp: 0.0,
            delay: 0.0,
            times: vec![0.0],
            shape: Arc::new(Shape(vec![0.0])),
            dur: 0.0,
        })),
        _ => None,
    }
}

impl AdcRaw {

    pub fn adc_num(&self) -> Option<u32> {

        Some(self.number_of_samples)
    }

    pub fn adc_dwell(&self) -> Option<f64> {
        Some(self.sample_time_us)
    }

    pub fn adc_freq(&self) -> Option<f64> {
        Some(0.0)
    }


    
}