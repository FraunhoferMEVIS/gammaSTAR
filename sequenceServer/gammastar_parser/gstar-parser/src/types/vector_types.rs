use super::scalar_types::{Moment, Sample};
// sample() types
#[derive(Debug, Clone)]
pub struct RfPulseSampleVec {
    pub amplitude: Vec<f64>,
    pub phase: Vec<f64>,
    pub frequency: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct GradientSampleVec {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct AdcBlockSampleVec {
    pub active: Vec<bool>,
    pub phase: Vec<f64>,
    pub frequency: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SampleVec {
    pub pulse: RfPulseSampleVec,
    pub gradient: GradientSampleVec,
    pub adc: AdcBlockSampleVec,
}

// integrate() types

#[derive(Debug, Clone)]
pub struct RfPulseMomentVec {
    pub angle: Vec<f64>,
    pub phase: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct GradientMomentVec {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MomentVec {
    pub pulse: RfPulseMomentVec,
    pub gradient: GradientMomentVec,
}

// Convert AoS to SoA

// use crate::{Moment, Sample};

impl From<Vec<Sample>> for SampleVec {
    fn from(value: Vec<Sample>) -> Self {
        let pulse = RfPulseSampleVec {
            amplitude: value.iter().map(|s| s.pulse.amplitude).collect(),
            phase: value.iter().map(|s| s.pulse.phase).collect(),
            frequency: value.iter().map(|s| s.pulse.frequency).collect(),
        };
        let gradient = GradientSampleVec {
            x: value.iter().map(|s| s.gradient.x).collect(),
            y: value.iter().map(|s| s.gradient.y).collect(),
            z: value.iter().map(|s| s.gradient.z).collect(),
        };
        let adc = AdcBlockSampleVec {
            active: value.iter().map(|s| s.adc.active).collect(),
            phase: value.iter().map(|s| s.adc.phase).collect(),
            frequency: value.iter().map(|s| s.adc.frequency).collect(),
        };

        Self {
            pulse,
            gradient,
            adc,
        }
    }
}

impl From<Vec<Moment>> for MomentVec {
    fn from(value: Vec<Moment>) -> Self {
        let pulse = RfPulseMomentVec {
            angle: value.iter().map(|s| s.pulse.angle).collect(),
            phase: value.iter().map(|s| s.pulse.phase).collect(),
        };
        let gradient = GradientMomentVec {
            x: value.iter().map(|s| s.gradient.x).collect(),
            y: value.iter().map(|s| s.gradient.y).collect(),
            z: value.iter().map(|s| s.gradient.z).collect(),
        };

        Self { pulse, gradient }
    }
}

// len() methods

impl RfPulseSampleVec {
    pub fn len(&self) -> usize {
        let len1 = self.amplitude.len();
        let len2 = self.phase.len();
        let len3 = self.frequency.len();
        assert!(len1 == len2 && len2 == len3);
        len1
    }
}

impl GradientSampleVec {
    pub fn len(&self) -> usize {
        let len1 = self.x.len();
        let len2 = self.y.len();
        let len3 = self.z.len();
        assert!(len1 == len2 && len2 == len3);
        len1
    }
}

impl AdcBlockSampleVec {
    pub fn len(&self) -> usize {
        let len1 = self.active.len();
        let len2 = self.phase.len();
        let len3 = self.frequency.len();
        assert!(len1 == len2 && len2 == len3);
        len1
    }
}

impl SampleVec {
    pub fn len(&self) -> usize {
        let len1 = self.pulse.len();
        let len2 = self.gradient.len();
        let len3 = self.adc.len();
        assert!(len1 == len2 && len2 == len3);
        len1
    }
}

impl RfPulseMomentVec {
    pub fn len(&self) -> usize {
        let len1 = self.angle.len();
        let len2 = self.phase.len();
        assert!(len1 == len2);
        len1
    }
}

impl GradientMomentVec {
    pub fn len(&self) -> usize {
        let len1 = self.x.len();
        let len2 = self.y.len();
        let len3 = self.z.len();
        assert!(len1 == len2 && len2 == len3);
        len1
    }
}

impl MomentVec {
    pub fn len(&self) -> usize {
        let len1 = self.pulse.len();
        let len2 = self.gradient.len();
        assert!(len1 == len2);
        len1
    }
}
