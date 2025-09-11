use std::fmt::Display;

// use crate::parse_file::Version;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ShapeDecompressionError {
    #[error("RLE count {value} is not integer at index {index}")]
    RleCountIsNotInteger { index: usize, value: f64 },
    #[error("Shape decompressed into {count} samples, expected {expected}")]
    WrongDecompressedCount { count: usize, expected: usize },
}

#[derive(Debug)]
pub enum EventType {
    Rf,
    Gx,
    Gy,
    Gz,
    Adc,
    Delay,
}

impl Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::Rf => "RF",
            EventType::Gx => "GX",
            EventType::Gy => "GY",
            EventType::Gz => "GZ",
            EventType::Adc => "ADC",
            EventType::Delay => "Delay",
        }
        .fmt(f)
    }
}

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("{ty} is too long for the containing block #{block_id}: {dur}s > {block_dur}s")]
    EventTooLong {
        ty: EventType,
        block_id: u32,
        dur: f64,
        block_dur: f64,
    },
    #[error("{ty} in block #{block_id} uses shapes with different sample counts: {length_1} vs {length_2}")]
    ShapeMismatch {
        ty: EventType,
        block_id: u32,
        length_1: usize,
        length_2: usize,
    },
    #[error("{ty} in block #{block_id} contains a negative timing: {timing}")]
    NegativeTiming {
        ty: EventType,
        block_id: u32,
        timing: f64,
    },
}

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Syntax error in pulseq file: {0}")]
    EzpcError(#[from] ezpc::EzpcError),
    #[error("Failed to parse float: {0}")]
    ParseFloat(#[from] std::num::ParseFloatError),
    #[error("Unsupported pulseq file version: {0}")]
    // UnsupportedVersion(Version),
    // #[error("Failed to decompress shape: {0}")]
    ShapeDecompressionError(#[from] ShapeDecompressionError),
}

#[derive(Debug)]
pub enum SectionType {
    Shapes,
    Delays,
    Adcs,
    Rfs,
    Gradients,
    Traps,
}

impl Display for SectionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SectionType::Shapes => "Shapes",
            SectionType::Delays => "Delays",
            SectionType::Adcs => "Adcs",
            SectionType::Rfs => "Rfs",
            SectionType::Gradients => "Gradients",
            SectionType::Traps => "Traps",
        }
        .fmt(f)
    }
}

#[derive(Error, Debug)]
pub enum ParseFovError {
    #[error(transparent)]
    ParseFloat(#[from] std::num::ParseFloatError),
    #[error("Expected 3 values, got {0}")]
    WrongValueCount(usize),
}

#[derive(Error, Debug)]
pub enum MissingDefinition {
    #[error(
        "Pulseq since 1.4 mandates time raster definitions, but is GradientRasterTime missing"
    )]
    GradientRasterTime,
    #[error("Pulseq since 1.4 mandates time raster definitions, but is RadiofrequencyRasterTime missing")]
    RadiofrequencyRasterTime,
    #[error("Pulseq since 1.4 mandates time raster definitions, but is AdcRasterTime missing")]
    AdcRasterTime,
    #[error(
        "Pulseq since 1.4 mandates time raster definitions, but is BlockDurationRaster missing"
    )]
    BlockDurationRaster,
}

// TODO: Include shape IDs into shapes for better error reporting

#[derive(Error, Debug)]
pub enum ConversionError {
    #[error("Expected a single [VERSION] section, found {0}")]
    VersionSectionCount(usize),
    #[error("{0} Section contains non-unique IDs")]
    EventIdReuse(SectionType),
    #[error("Found re-used IDs between Trap and Gradient events")]
    GradTrapIdReuse,
    #[error("Definitions contain non-unique keys")]
    NonUniqueDefinition,
    #[error("Referenced {ty} with id {id} does not exist")]
    BrokenRef { ty: EventType, id: u32 },
    #[error(transparent)]
    MissingDefinition(#[from] MissingDefinition),
    #[error("Failed to parse FOV: {0}")]
    ParseFovError(#[from] ParseFovError),
    #[error(transparent)]
    ParseFloat(#[from] std::num::ParseFloatError),
    #[error("Shape with index {0} does not exist")]
    ShapeNotFound(u32),
    #[error("Can't use 0 as shape index")]
    ShapeIndexZero,
    #[error("Used a shape of length {shape_len} together with a time shape of length {time_len}")]
    TimeShapeMismatch { shape_len: usize, time_len: usize },
    #[error("Used a shape as time shape which contained non-integer values.")]
    TimeShapeNonInteger,
    #[error("Used a shape as time shape which is not strictly increasing")]
    TimeShapeNonIncreasing,
}

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    ParseError(#[from] ParseError),
    #[error("Sequence validation failed: {0}")]
    ValidationError(#[from] ValidationError),
    #[error("Failed to convert parsed file into sequence: {0}")]
    ConversionError(#[from] ConversionError),
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}
