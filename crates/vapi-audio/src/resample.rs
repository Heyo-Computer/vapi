//! Getting arbitrary input to the 16 kHz mono the model expects.

/// Average interleaved channels down to one.
pub fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect()
}

/// Linear resampling to `target` Hz.
///
/// Linear rather than windowed-sinc, which is a real approximation and worth
/// naming: it attenuates the top of the band and lets a little aliasing
/// through when downsampling. Speech energy for this model lives below 8 kHz
/// and most callers send 16 kHz already, so the common path does no
/// resampling at all; a caller sending 44.1 kHz music would want better.
pub fn resample_linear(samples: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || samples.is_empty() || from == 0 {
        return samples.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let n = ((samples.len() as f64) / ratio).floor() as usize;
    (0..n)
        .map(|i| {
            let at = i as f64 * ratio;
            let left = at.floor() as usize;
            let frac = (at - left as f64) as f32;
            let a = samples[left.min(samples.len() - 1)];
            let b = samples[(left + 1).min(samples.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_rates_are_left_alone() {
        let s = vec![0.1, 0.2, 0.3];
        assert_eq!(resample_linear(&s, 16_000, 16_000), s);
    }

    #[test]
    fn downsampling_halves_the_length() {
        let s: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let out = resample_linear(&s, 32_000, 16_000);
        assert_eq!(out.len(), 500);
        // A ramp stays a ramp, at twice the step.
        assert!((out[10] - 20.0).abs() < 1e-3, "{}", out[10]);
    }

    #[test]
    fn upsampling_interpolates_between_neighbours() {
        let out = resample_linear(&[0.0, 1.0], 8_000, 16_000);
        assert_eq!(out.len(), 4);
        assert!((out[1] - 0.5).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn channels_are_averaged_not_dropped() {
        // Taking the left channel would silently lose a speaker recorded on
        // the right.
        let stereo = [1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        assert_eq!(to_mono(&stereo, 2), vec![0.5, 0.5, 0.5]);
        assert_eq!(to_mono(&stereo, 1), stereo.to_vec());
    }
}
