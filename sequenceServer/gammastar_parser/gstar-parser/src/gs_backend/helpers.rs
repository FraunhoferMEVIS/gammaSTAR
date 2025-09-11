use crate::gs_sequence::{Gradient, Rf, Shape};

use crate::util::{Rotation, Spin};

pub fn integrate_grad(
    gx: &Gradient,
    t_start: f64,
    t_end: f64,
    block_start: f64,
    grad_raster: f64,
) -> f64 {
    match gx {
        Gradient::Free { amp, delay, shape, ..} => {
            //TODO : check if this works without amp
            //It does work without amp
            integrate_free(
            // amp * integrate_free(
                t_start - block_start - delay,
                t_end - block_start - delay,
                shape,
                grad_raster,
            )
        }
        Gradient::Trap {
            amp,
            rise,
            flat,
            fall,
            delay,
            ..
        } => {
            amp * integrate_trap(
                t_start - block_start - delay,
                t_end - block_start - delay,
                *rise,
                *flat,
                *fall,
            )
        }
    }
}

// TODO: change spin + rotation matrix to a unified rotation struct (matrix or quaternion etc.)
// that is returned from this function
pub fn integrate_rf(
    rf: &Rf,
    spin: &mut Spin,
    t_start: f64,
    t_end: f64,
    block_start: f64,
    rf_raster: f64,
) {
    for i in 0..rf.amp_shape.0.len() {
        let dwell = rf_raster;
        // Start time of the sample number i
        let t = block_start + rf.delay + i as f64 * dwell;

        // Skip samples before t_start, quit when reaching t_end
        if t + dwell < t_start {
            continue;
        }
        if t_end <= t {
            break;
        }

        // We could do the clamping for all samples, but when integrating
        // over many samples, it seems to be very sensitive to accumulating
        // errors. Only doing it in the edge cases is much more robust.
        let dur = if t_start <= t && t + dwell <= t_end {
            dwell
        } else {
            // Clamp the sample interval to the integration interval
            let t0 = t.clamp(t_start, t_end);
            let t1 = (t + dwell).clamp(t_start, t_end);
            t1 - t0
        };

        *spin *= Rotation::new(
            rf.amp * rf.amp_shape.0[i] * dur * std::f64::consts::TAU,
            rf.phase + rf.phase_shape.0[i] * std::f64::consts::TAU,
        );
    }
}

pub fn sample_grad(t: f64, grad: &Gradient, grad_raster: f64) -> f64 {
    match grad {
        Gradient::Free { amp, delay, shape , ..} => {
            let index = ((t - delay) / grad_raster - 0.5).ceil() as usize;
            shape.0.get(index).map_or(0.0, |x| 1.0 * x)
        }
        Gradient::Trap {
            amp,
            rise,
            flat,
            fall,
            delay,
            ..
        } =>  trap_sample(t - delay, *rise, *flat, *fall),
    }
}

pub fn trap_sample(t: f64, rise: f64, flat: f64, fall: f64) -> f64 {
    if t < 0.0 {
        0.0
    } else if t < rise {
        t / rise
    } else if t < rise + flat {
        1.0
    } else if t < rise + flat + fall {
        ((rise + flat + fall) - t) / fall
    } else {
        0.0
    }
}

pub fn integrate_trap(t_start: f64, t_end: f64, rise: f64, flat: f64, fall: f64) -> f64 {
    let integral = |t| {
        if t <= rise {
            0.5 * t * t / rise
        } else if t <= rise + flat {
            (0.5 * rise) + (t - rise)
        } else {
            let rev_t = rise + flat + fall - t;
            (0.5 * rise) + flat + (0.5 * (fall - rev_t * rev_t / fall))
        }
    };
    let t_min = 0.0;
    let t_max = rise + flat + fall;
    integral(t_end.clamp(t_min, t_max)) - integral(t_start.clamp(t_min, t_max))
}

pub fn integrate_free(t_start: f64, t_end: f64, shape: &Shape, dwell: f64) -> f64 {
    let mut integrated = 0.0;

    for i in 0..shape.0.len() {
        // Start time of the sample number i
        let t = i as f64 * dwell;

        // Skip samples before t_start, quit when reaching t_end
        if t + dwell <= t_start {
            continue;
        }
        if t_end <= t {
            break;
        }

        // We could do the clamping for all samples, but when integrating
        // over many samples, it seems to be very sensitive to accumulating
        // errors. Only doing it in the edge cases is much more robust.
        let dur = if t_start <= t && t + dwell <= t_end {
            dwell
        } else {
            // Clamp the sample interval to the integration interval
            let t0 = t.clamp(t_start, t_end);
            let t1 = (t + dwell).clamp(t_start, t_end);
            t1 - t0
        };

        integrated += shape.0[i] * dur;
    }

    integrated
}
