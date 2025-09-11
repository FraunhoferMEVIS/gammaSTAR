// TODO: implement From trait for the reverse operation (see vector_types)

// sample() types

/// Contains the RF Pulse state for a single point in time.
#[derive(Default, Debug, Clone, Copy)]
pub struct RfPulseSample {
    /// Unit: `Hz`
    pub amplitude: f64,
    /// Unit: `rad`
    pub phase: f64,
    /// Unit: `Hz`
    pub frequency: f64,
}

/// Contains the gradient amplitudes for a single point in time.
#[derive(Default, Debug, Clone, Copy)]
pub struct GradientSample {
    /// Unit: `Hz / m`
    pub x: f64,
    /// Unit: `Hz / m`
    pub y: f64,
    /// Unit: `Hz / m`
    pub z: f64,
}

/// Contains the ADC state for a single point in time. NOTE: this does not
/// indicate if a particular time point is sampled, only that an ADC block is
/// active (or not) at the particular point in time. Use the sequence POI API
/// to fetch the ADC sample locations.
#[derive(Default, Debug, Clone, Copy)]
pub struct AdcBlockSample {
    /// Specifies if the ADC is active, not if this is an ADC sample
    pub active: bool,
    /// Unit: `rad`
    pub phase: f64,
    /// Unit: `Hz`
    pub frequency: f64,
}

/// See `RfPulseSample`, `GradientSample` and `AdcBlockSample`
#[derive(Default, Debug, Clone, Copy)]
pub struct Sample {
    pub pulse: RfPulseSample,
    pub gradient: GradientSample,
    pub adc: AdcBlockSample,
}

// integrate() types

/// Resulting gradient moments by integrating gradients over some time period.
#[derive(Default, Debug, Clone, Copy)]
pub struct GradientMoment {
    /// Unit: `rad / m`
    pub x: f64,
    /// Unit: `rad / m`
    pub y: f64,
    /// Unit: `rad / m`
    pub z: f64,
}

/// Resulting flip angle by integrating an RF pulse over some time period.
#[derive(Default, Debug, Clone, Copy)]
pub struct RfPulseMoment {
    /// Unit: `rad`
    pub angle: f64,
    /// Unit: `rad`
    pub phase: f64,
}

/// See `RfPulseMoment` and `GradientMoment`
#[derive(Default, Debug, Clone, Copy)]
pub struct Moment {
    pub pulse: RfPulseMoment,
    pub gradient: GradientMoment,
}
