
mod scalar_types;
mod vector_types;
pub use scalar_types::*;
pub use vector_types::*;

/// Used for Block::Gradient(channel)
#[derive(Debug, Clone, Copy)]
pub enum GradientChannel {
    X,
    Y,
    Z,
}

/// Used to fetch the next POI or block time span of the given type.
#[derive(Debug, Clone, Copy)]
pub enum EventType {
    RfPulse,
    Adc,
    Gradient(GradientChannel),
}
