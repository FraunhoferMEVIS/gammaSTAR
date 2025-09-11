
use serde_json::error;
use crate::gs_sequence::Gradient;
// use crate::gstar_file::*;
use crate::{ gs_sequence as sequence, util};
use crate::traits::Backend;
use crate::types::*;
mod helpers;
use std::path::Path;
pub struct GstarSequence {
    pub blocks: Vec<(f64, sequence::NestedBlock)>,
    pub raster: f64,
    pub fov: Option<(f64, f64, f64)>,
}


impl GstarSequence {

    pub fn calculate_durations(&self) -> Vec<f64> {
        self.blocks
            .iter()
            .enumerate()
            .map(|(i, (_, current_block))| {
                if i < self.blocks.len() - 1 {
                    let next_block_start_time = self.blocks[i + 1].1.tstart;
                    next_block_start_time - current_block.tstart
                } else {
                    // Handle last block duration as needed
                    current_block.duration  // Placeholder
                }
            })
            .collect()
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, error::Error>{
        let seq = sequence::Sequence::form_file(path);
        Ok(Self::from_seq(seq))
    }

    fn from_seq(seq: sequence::Sequence) -> Self {
        
        let blocks: Vec<(f64, sequence::NestedBlock)> = seq
            .blocks
            .into_iter()
            .scan(0.0, |t_start, block| {
                
                // let tmp = *t_start;
                //TODO: check if this is p2p duration or without delays
                // *t_start += block.duration as f64;
                Some((block.tstart, block))
            })
            .collect();
        // Iterate over blocks to update the duration
        
        // We could check for e.g. lower case fov and if definition is in mm
        let fov = seq
            .fov
;       
        let mut self_sequence = Self {
            blocks,
            raster: seq.time_raster,
            fov: seq.fov,
        };
        let durations = self_sequence.calculate_durations();
        for (block, duration) in self_sequence.blocks.iter_mut().zip(durations.iter()) {
            block.1.duration = *duration;
        }

        self_sequence
        // Self {
        //     blocks,
        //     raster: seq.time_raster,
        //     fov,
        // }
    }

}   

impl Backend for GstarSequence {
    fn fov(&self) -> Option<(f64, f64, f64)> {
        self.fov
    }

    fn duration(&self) -> f64 {
        self.blocks.iter().map(|(_, b)| b.duration as f64).sum()
    }
    // #[cfg(feature = "debug")]
    fn events(&self, ty: EventType, t_start: f64, t_end: f64, max_count: usize) -> Vec<f64> {
        // NOTE: The indirection by using a trait object seems to be neglectable in terms of
        // performance, although it makes the API a bit worse, as the time range that is
        // usually only constructed for the function call now needs a reference.
        let mut t = t_start;
        let mut pois = Vec::new();
        // TODO: this currently is based on the PulseqSequence::next_poi function.
        // Replace with a more efficient impl that directly fetches a list of samples
        while let Some(t_next) = self.next_poi(t, ty) {
            // Important: make t_end exclusive so we don't need to advance by some small value
            if t_next >= t_end || pois.len() >= max_count {
                break;
            }
            pois.push(t_next);
            t = t_next + 1e-9;
        }

        pois
    }
    // #[cfg(feature = "debug")]
    fn encounter(&self, t_start: f64, ty: EventType) -> Option<(f64, f64)> {
        let idx_start = match self
            .blocks
            .binary_search_by(|(block_start, _)| block_start.total_cmp(&t_start))
        {
            Ok(idx) => idx,             // start with the exact match
            Err(idx) => idx.max(1) - 1, // start before the insertion point
        };

        for (block_start, block) in &self.blocks[idx_start..] {
            let t = match ty {
                EventType::RfPulse => block
                    .rf
                    .as_ref()
                    .map(|rf| (rf.delay, rf.rf_dur)),
                EventType::Adc => block.adc.as_ref().map(|adc| (adc.delay, adc.adc_dur)),
                EventType::Gradient(channel) => {
                    let gradient = match channel {
                        GradientChannel::X => block.gx.as_ref(),
                        GradientChannel::Y => block.gy.as_ref(),
                        GradientChannel::Z => block.gz.as_ref(),
                    };
                
                    gradient.map(|grad_arc| {
                        let grad = grad_arc.as_ref(); // Dereference the Arc to get to the Gradient
                        match grad {
                            Gradient::Free { delay, dur, .. } => (*delay, *dur),
                            Gradient::Trap { delay, dur, .. } => (*delay, *dur),
                        }
                    })
                },
            };

            if let Some((delay, dur)) = t {
                if block_start + delay >= t_start {
                    return Some((block_start + delay, block_start + dur));
                }
            }
        }

        None
    }
    // #[cfg(feature = "debug")]
    fn integrate(&self, time: &[f64]) -> Vec<Moment> {
        let mut moments = Vec::new();
        for t in time.windows(2) {
            let (pulse, gradient) = self.integrate(t[0], t[1]);
            moments.push(Moment { pulse, gradient });
        }
        moments
    }
    // #[cfg(feature = "debug")]
    fn sample(&self, time: &[f64]) -> Vec<Sample> {
        time.into_iter()
            .map(|t| {
                let (pulse, gradient, adc) = self.sample(*t);
                Sample {
                    pulse,
                    gradient,
                    adc,
                }
            })
            .collect()
    }
}
// #[cfg(feature = "debug")]
impl GstarSequence {
    fn next_poi(&self, t_start: f64, ty: EventType) -> Option<f64> {
        let idx_start = match self
            .blocks
            .binary_search_by(|(block_start, _)| block_start.total_cmp(&t_start))
        {
            Ok(idx) => idx,             // start with the exact match
            Err(idx) => idx.max(1) - 1, // start before the insertion point
        };

        for (block_start, block) in &self.blocks[idx_start..] {
            // We sample in between samples, so for e.g., a shape of len=10
            // there will be 0..=10 -> 11 samples.
            let t = t_start - block_start;
            let t = match ty {
                EventType::RfPulse => block.rf.as_ref().map(|rf| {
                    let idx = ((t - rf.delay) / self.raster)
                        .clamp(0.0, rf.amp_shape.0.len() as f64 - 1.0)
                        .ceil();
                    rf.delay + idx * self.raster
                }),
                EventType::Adc => block.adc.as_ref().map(|adc| {
                    // Here we actually sample in the centers instead of edges because,
                    // well, that's where the ADC samples are!
                    // TODO: check what does adc delay actually stands for.
                    let idx = ((t - adc.delay) / adc.dwell - 0.5)
                        .clamp(0.0, adc.num as f64 - 1.0)
                        .ceil();
                    adc.delay + (idx + 0.5) * adc.dwell
                }),
                
                EventType::Gradient(channel) => match channel {
                    GradientChannel::X => block.gx.as_ref(),
                    GradientChannel::Y => block.gy.as_ref(),
                    GradientChannel::Z => block.gz.as_ref(),
                }
                .map(|grad| match grad.as_ref() {
                    Gradient::Free { delay, shape, .. } => {
                        let idx = ((t - delay) / self.raster)
                            .clamp(0.0, shape.0.len() as f64)
                            .ceil();
                        delay + idx * self.raster
                    }
                    &Gradient::Trap {
                        rise,
                        flat,
                        fall,
                        delay,
                        ..
                    } => {
                        // The four vertices of the trap are its POIs
                        if t < delay {
                            delay
                        } else if t < rise {
                            delay + rise
                        } else if t < rise + flat {
                            delay + rise + flat
                        } else {
                            // No if bc. of check below and mandatory else branch
                            delay + rise + flat + fall
                        }
                    }
                }),
            };

            if let Some(t) = t {
                if t + block_start >= t_start {
                    return Some(t + block_start);
                }
            }
        }

        None
    }

    fn integrate(&self, mut t_start: f64, mut t_end: f64) -> (RfPulseMoment, GradientMoment) {
        let mut sign = 1.0;
        if t_end < t_start {
            // Integrate backwards and flip sign
            std::mem::swap(&mut t_start, &mut t_end);
            sign = -1.0;
        }

        let idx_start = match self
            .blocks
            .binary_search_by(|(block_start, _)| block_start.total_cmp(&t_start))
        {
            Ok(idx) => idx,             // start with the exact match
            Err(idx) => idx.max(1) - 1, // start before the insertion point
        };

        let mut spin = util::Spin::relaxed();
        let mut grad = GradientMoment {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        for (block_start, block) in &self.blocks[idx_start..] {
            if *block_start >= t_end {
                break;
            }
            if let Some(gx) = block.gx.as_ref() {
                grad.x += helpers::integrate_grad(
                    gx.as_ref(),
                    t_start,
                    t_end,
                    *block_start,
                    self.raster,
                );
            }
            if let Some(gy) = block.gy.as_ref() {
                grad.y += helpers::integrate_grad(
                    gy.as_ref(),
                    t_start,
                    t_end,
                    *block_start,
                    self.raster,
                );
            }
            if let Some(gz) = block.gz.as_ref() {
                grad.z += helpers::integrate_grad(
                    gz.as_ref(),
                    t_start,
                    t_end,
                    *block_start,
                    self.raster,
                );
            }
            if let Some(rf) = block.rf.as_ref() {
                helpers::integrate_rf(rf, &mut spin, t_start, t_end, *block_start, self.raster);
            }
        }

        (
            RfPulseMoment {
                angle: sign * spin.angle(),
                phase: sign * spin.phase(),
            },
            GradientMoment {
                x: sign * grad.x,
                y: sign * grad.y,
                z: sign * grad.z,
            },
        )
    }
    // #[cfg(feature = "debug")]
    fn sample(&self, t: f64) -> (RfPulseSample, GradientSample, AdcBlockSample) {
        let block_idx = match self
            .blocks
            .binary_search_by(|(block_start, _)| block_start.total_cmp(&t))
        {
            Ok(idx) => idx,             // sample is exactly at beginning of block
            Err(idx) => idx.max(1) - 1, // sample is somewhere in the block
        };
        let (block_start, block) = &self.blocks[block_idx];

        let pulse_sample = if let Some(rf) = &block.rf {
            let index = ((t - block_start - rf.delay) / self.raster - 0.5).ceil() as usize;
            if index < rf.amp_shape.0.len() {
                RfPulseSample {
                    amplitude: rf.amp * rf.amp_shape.0[index],
                    phase: rf.phase + rf.phase_shape.0[index] * std::f64::consts::TAU,
                    frequency: rf.freq,
                }
            } else {
                RfPulseSample::default()
            }
        } else {
            RfPulseSample::default()
        };

        let x = block.gx.as_ref().map_or(0.0, |gx| {
            helpers::sample_grad(t - block_start, gx.as_ref(), self.raster)
        });
        let y = block.gy.as_ref().map_or(0.0, |gy| {
            helpers::sample_grad(t - block_start, gy.as_ref(), self.raster)
        });
        let z = block.gz.as_ref().map_or(0.0, |gz| {
            helpers::sample_grad(t - block_start, gz.as_ref(), self.raster)
        });

        let adc_sample = if let Some(adc) = &block.adc {
            if block_start + adc.delay <= t
                && t <= block_start + adc.delay + adc.num as f64 * adc.dwell
            {
                AdcBlockSample {
                    active: true,
                    phase: adc.phase,
                    frequency: adc.freq,
                }
            } else {
                AdcBlockSample::default()
            }
        } else {
            AdcBlockSample::default()
        };

        (pulse_sample, GradientSample { x, y, z }, adc_sample)
    }
}
