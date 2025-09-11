use pyo3::pyclass;

// sample() types

#[pyclass]
#[derive(Clone, Debug)]
pub struct RfPulseSampleVec {
    #[pyo3(get)]
    pub amplitude: Vec<f64>,
    #[pyo3(get)]
    pub phase: Vec<f64>,
    #[pyo3(get)]
    pub frequency: Vec<f64>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct GradientSampleVec {
    #[pyo3(get)]
    pub x: Vec<f64>,
    #[pyo3(get)]
    pub y: Vec<f64>,
    #[pyo3(get)]
    pub z: Vec<f64>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct AdcBlockSampleVec {
    #[pyo3(get)]
    pub active: Vec<bool>,
    #[pyo3(get)]
    pub phase: Vec<f64>,
    #[pyo3(get)]
    pub frequency: Vec<f64>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct SampleVec {
    #[pyo3(get)]
    pub pulse: RfPulseSampleVec,
    #[pyo3(get)]
    pub gradient: GradientSampleVec,
    #[pyo3(get)]
    pub adc: AdcBlockSampleVec,
}

// integrate() types

#[pyclass]
#[derive(Clone, Debug)]
pub struct RfPulseMomentVec {
    #[pyo3(get)]
    pub angle: Vec<f64>,
    #[pyo3(get)]
    pub phase: Vec<f64>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct GradientMomentVec {
    #[pyo3(get)]
    pub x: Vec<f64>,
    #[pyo3(get)]
    pub y: Vec<f64>,
    #[pyo3(get)]
    pub z: Vec<f64>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct MomentVec {
    #[pyo3(get)]
    pub pulse: RfPulseMomentVec,
    #[pyo3(get)]
    pub gradient: GradientMomentVec,
}