use crate::{gs_backend, gs_sequence, sequence, types::{EventType, Moment, Sample}};
use std::path::Path;
use sequence::Sequence;
use serde_json::error;

pub fn load_gstar<P: AsRef<Path>>(path: P) -> Result<Sequence, error::Error> {
    Ok(Sequence(Box::new(gs_backend::GstarSequence::load(
        path,
    )?)))
}

pub trait Backend: Send {
    /// Return the FOV of the Sequence, if it is available
    fn fov(&self) -> Option<(f64, f64, f64)>;

    /// Duration of the MRI sequence: no samples, blocks, etc. exist outside
    /// of the time range [0, duration()]
    fn duration(&self) -> f64;
    // #[cfg(feature = "debug")]
    /// Returns all events of the given type in the given duration.
    /// t_start is inclusive, t_end is exclusive. If a max_count is given and
    /// reached, there might be more events in the time span that are not returned.
    fn events(&self, ty: EventType, t_start: f64, t_end: f64, max_count: usize) -> Vec<f64>;
    // #[cfg(feature = "debug")]
    /// Returns the time range of the next encounter of the given type.
    /// If `t_start` is inside of a block, this block is not returned: only
    /// blocks **starting** after (or exactly on) `t_start` are considered.
    /// TODO: EventType should be the first parameter
    fn encounter(&self, t_start: f64, ty: EventType) -> Option<(f64, f64)>;
    // #[cfg(feature = "debug")]
    /// Samples the sequence at the given time points
    fn sample(&self, time: &[f64]) -> Vec<Sample>;
    // #[cfg(feature = "debug")]
    /// Integrates over the n-1 time intervals given by the list of n time points.
    fn integrate(&self, time: &[f64]) -> Vec<Moment>;


}
