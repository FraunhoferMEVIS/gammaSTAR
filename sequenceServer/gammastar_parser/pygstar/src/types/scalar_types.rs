use pyo3::pyclass;

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct RfPulseSample {
    #[pyo3(get)]
    pub amplitude: f64,
    #[pyo3(get)]
    pub phase: f64,
    #[pyo3(get)]
    pub frequency: f64,
}

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct GradientSample {
    #[pyo3(get)]
    pub x: f64,
    #[pyo3(get)]
    pub y: f64,
    #[pyo3(get)]
    pub z: f64,
}

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct AdcBlockSample {
    #[pyo3(get)]
    pub active: bool,
    #[pyo3(get)]
    pub phase: f64,
    #[pyo3(get)]
    pub frequency: f64,
}

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    #[pyo3(get)]
    pub pulse: RfPulseSample,
    #[pyo3(get)]
    pub gradient: GradientSample,
    #[pyo3(get)]
    pub adc: AdcBlockSample,
}

// integrate() types

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct GradientMoment {
    #[pyo3(get)]
    pub x: f64,
    #[pyo3(get)]
    pub y: f64,
    #[pyo3(get)]
    pub z: f64,
}

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct RfPulseMoment {
    #[pyo3(get)]
    pub angle: f64,
    #[pyo3(get)]
    pub phase: f64,
}

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct Moment {
    #[pyo3(get)]
    pub pulse: RfPulseMoment,
    #[pyo3(get)]
    pub gradient: GradientMoment,
}
