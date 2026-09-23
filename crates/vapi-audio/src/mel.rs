//! Short-time Fourier transform and mel filterbank.

use std::f32::consts::PI;

/// Everything the frontend needs, read from the model's processor config.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MelSettings {
    pub sampling_rate: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    /// Upper edge of the filterbank. Half the sampling rate unless the model
    /// says otherwise.
    pub max_frequency: f32,
    /// The log floor: values below `global_log_mel_max - DYNAMIC_RANGE` are
    /// clamped. A constant rather than the utterance's own maximum, which is
    /// what makes the frontend streamable.
    pub global_log_mel_max: f32,
}

impl Default for MelSettings {
    fn default() -> Self {
        Self {
            sampling_rate: 16_000,
            n_fft: 400,
            win_length: 400,
            hop_length: 160,
            n_mels: 128,
            max_frequency: 8_000.0,
            global_log_mel_max: 1.5,
        }
    }
}

impl MelSettings {
    /// Read `processor_config.json`'s `feature_extractor` block, keeping the
    /// defaults for anything it does not mention.
    pub fn from_json(text: &str) -> vapi_core::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| vapi_core::Error::Config(format!("processor config: {e}")))?;
        let fe = v.get("feature_extractor").unwrap_or(&v);
        let d = Self::default();
        let num = |key: &str, fallback: usize| {
            fe.get(key)
                .and_then(serde_json::Value::as_u64)
                .map_or(fallback, |n| n as usize)
        };
        let sampling_rate = num("sampling_rate", d.sampling_rate);
        Ok(Self {
            sampling_rate,
            n_fft: num("n_fft", d.n_fft),
            win_length: num("win_length", d.win_length),
            hop_length: num("hop_length", d.hop_length),
            n_mels: num("feature_size", d.n_mels),
            max_frequency: sampling_rate as f32 / 2.0,
            global_log_mel_max: fe
                .get("global_log_mel_max")
                .and_then(serde_json::Value::as_f64)
                .map_or(d.global_log_mel_max, |x| x as f32),
        })
    }

    /// Frequency bins a real FFT of this size produces.
    pub fn num_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Seconds of audio one frame advances by.
    pub fn frame_seconds(&self) -> f32 {
        self.hop_length as f32 / self.sampling_rate as f32
    }
}

/// How far below the global maximum the log scale is floored.
const DYNAMIC_RANGE: f32 = 8.0;

fn hz_to_mel(hz: f32) -> f32 {
    // Slaney: linear below 1 kHz, logarithmic above. Not the HTK formula;
    // they differ by a few percent per filter, which is enough to move every
    // mel bin and nothing like enough to look broken.
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    let logstep = (6.4f32).ln() / 27.0;
    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
    } else {
        3.0 * hz / 200.0
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    let logstep = (6.4f32).ln() / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * logstep).exp()
    } else {
        200.0 * mel / 3.0
    }
}

/// Triangular mel filters, Slaney-normalised, as `[n_mels][num_bins]`.
///
/// The triangles are laid out in **Hz**, not in mel. Both conventions are in
/// use — `mel_filter_bank(triangularize_in_mel_space=...)` in the reference
/// library — and they are not close: laying them out in mel space here moves
/// every bin by up to 2.1, against a total range of 2.
fn filterbank(s: &MelSettings) -> Vec<Vec<f32>> {
    let bins = s.num_bins();
    let fft_freqs: Vec<f32> = (0..bins)
        .map(|i| i as f32 * (s.sampling_rate as f32 / 2.0) / (bins - 1) as f32)
        .collect();

    let lo = hz_to_mel(0.0);
    let hi = hz_to_mel(s.max_frequency);
    let points: Vec<f32> = (0..s.n_mels + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f32 / (s.n_mels + 1) as f32))
        .collect();

    let mut out = vec![vec![0f32; bins]; s.n_mels];
    for m in 0..s.n_mels {
        let (left, centre, right) = (points[m], points[m + 1], points[m + 2]);
        // Slaney normalisation: every filter integrates to the same area, so
        // a wide high-frequency filter does not drown a narrow low one.
        let scale = 2.0 / (right - left);
        for (b, &f) in fft_freqs.iter().enumerate() {
            let up = (f - left) / (centre - left);
            let down = (right - f) / (right - centre);
            out[m][b] = up.min(down).max(0.0) * scale;
        }
    }
    out
}

/// The centre frequency of a mel filter, in Hz. For tests, and for saying
/// something useful in a log line.
pub fn filter_centre_hz(s: &MelSettings, filter: usize) -> f32 {
    let lo = hz_to_mel(0.0);
    let hi = hz_to_mel(s.max_frequency);
    mel_to_hz(lo + (hi - lo) * (filter + 1) as f32 / (s.n_mels + 1) as f32)
}

/// Periodic Hann window, which is what `torch.hann_window` gives by default.
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

/// A configured frontend: windows and filters built once.
pub struct MelSpectrogram {
    settings: MelSettings,
    window: Vec<f32>,
    filters: Vec<Vec<f32>>,
    /// Twiddles for a direct real-input DFT, `[bin][sample]`.
    ///
    /// A DFT rather than an FFT because 400 is not a power of two and only
    /// 201 bins are wanted: 201 × 400 multiply-adds per frame at 100 frames a
    /// second is about 8 M operations a second per stream, which is nothing,
    /// and a mixed-radix FFT here would be a hundred lines of arithmetic that
    /// has to be exactly right.
    cos_table: Vec<f64>,
    sin_table: Vec<f64>,
}

impl MelSpectrogram {
    pub fn new(settings: MelSettings) -> Self {
        let bins = settings.num_bins();
        let n = settings.n_fft;
        let mut cos_table = vec![0f64; bins * n];
        let mut sin_table = vec![0f64; bins * n];
        for k in 0..bins {
            for t in 0..n {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
                cos_table[k * n + t] = angle.cos();
                sin_table[k * n + t] = angle.sin();
            }
        }
        Self {
            window: hann(settings.win_length),
            filters: filterbank(&settings),
            settings,
            cos_table,
            sin_table,
        }
    }

    pub fn settings(&self) -> &MelSettings {
        &self.settings
    }

    /// How many frames `samples` of audio produces.
    ///
    /// Centred frames, then the last one dropped: with reflect padding a
    /// signal of `n` samples gives `1 + n / hop` frames, and the reference
    /// discards the final one because it reaches past the end of the audio.
    pub fn num_frames(&self, samples: usize) -> usize {
        samples / self.settings.hop_length
    }

    /// Log-mel features as `[n_mels][frames]`, the layout the model wants.
    pub fn compute(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let s = &self.settings;
        let frames = self.num_frames(samples.len());
        let mut out = vec![vec![0f32; frames]; s.n_mels];
        if frames == 0 {
            return out;
        }

        let pad = s.n_fft / 2;
        let mut frame = vec![0f64; s.n_fft];
        let mut power = vec![0f64; s.num_bins()];
        let floor = s.global_log_mel_max - DYNAMIC_RANGE;

        #[allow(clippy::needless_range_loop)] // `t` indexes every mel row, not one slice.
        for t in 0..frames {
            // Reflect padding, computed per sample rather than by building a
            // padded copy of a three-hour recording.
            let start = t * s.hop_length;
            for (i, slot) in frame.iter_mut().enumerate() {
                let at = start + i;
                let idx = reflect(at as isize - pad as isize, samples.len());
                *slot = samples[idx] as f64 * self.window[i.min(self.window.len() - 1)] as f64;
            }
            self.power_spectrum(&frame, &mut power);
            for (m, filter) in self.filters.iter().enumerate() {
                let energy: f64 = filter.iter().zip(&power).map(|(w, p)| *w as f64 * p).sum();
                let log = (energy.max(1e-10).log10() as f32).max(floor);
                out[m][t] = (log + 4.0) / 4.0;
            }
        }
        out
    }

    /// Accumulated in f64 on purpose. High-frequency energy in speech sits
    /// orders of magnitude below the fundamental, and an f32 sum of 400 terms
    /// loses it to cancellation: it left the bottom of the filterbank exact
    /// and the top 2.6% out.
    fn power_spectrum(&self, frame: &[f64], out: &mut [f64]) {
        let n = self.settings.n_fft;
        for (k, slot) in out.iter_mut().enumerate() {
            let (mut re, mut im) = (0f64, 0f64);
            let base = k * n;
            for (t, &x) in frame.iter().enumerate() {
                re += x * self.cos_table[base + t];
                im += x * self.sin_table[base + t];
            }
            *slot = re * re + im * im;
        }
    }
}

/// Index into `samples` for a position that may fall outside it, reflecting
/// at both edges as `torch.nn.functional.pad(mode="reflect")` does.
fn reflect(at: isize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    if len == 1 {
        return 0;
    }
    let period = 2 * (len as isize - 1);
    let mut i = at.rem_euclid(period);
    if i >= len as isize {
        i = period - i;
    }
    i as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_mirrors_at_both_edges() {
        // 0 1 2 3 reflected: ... 2 1 | 0 1 2 3 | 2 1 ...
        assert_eq!(reflect(-1, 4), 1);
        assert_eq!(reflect(-2, 4), 2);
        assert_eq!(reflect(-3, 4), 3);
        assert_eq!(reflect(0, 4), 0);
        assert_eq!(reflect(3, 4), 3);
        assert_eq!(reflect(4, 4), 2);
        assert_eq!(reflect(5, 4), 1);
        assert_eq!(reflect(0, 1), 0);
    }

    #[test]
    fn the_filterbank_covers_the_band_without_gaps() {
        let s = MelSettings::default();
        let fb = filterbank(&s);
        assert_eq!(fb.len(), 128);
        assert_eq!(fb[0].len(), 201);
        // Every filter has weight somewhere, and none reaches past Nyquist.
        for (m, f) in fb.iter().enumerate() {
            assert!(f.iter().any(|&w| w > 0.0), "filter {m} is empty");
            assert!(f.iter().all(|&w| w >= 0.0));
        }
        // Slaney normalisation: higher filters are wider and so are shorter.
        let peak = |f: &Vec<f32>| f.iter().cloned().fold(0f32, f32::max);
        assert!(peak(&fb[0]) > peak(&fb[127]));
    }

    #[test]
    fn a_frame_is_produced_for_every_hop() {
        let mel = MelSpectrogram::new(MelSettings::default());
        assert_eq!(mel.num_frames(16_000), 100, "one second is 100 frames");
        assert_eq!(mel.num_frames(19_200), 120);
        assert_eq!(mel.num_frames(0), 0);
    }

    #[test]
    fn silence_sits_on_the_floor() {
        // With a global floor rather than a per-utterance maximum, silence
        // has one fixed value — the property that makes this streamable.
        let s = MelSettings::default();
        let mel = MelSpectrogram::new(s);
        let out = mel.compute(&vec![0f32; 16_000]);
        let want = (s.global_log_mel_max - DYNAMIC_RANGE + 4.0) / 4.0;
        for row in &out {
            for &x in row {
                assert!((x - want).abs() < 1e-6, "{x} vs {want}");
            }
        }
    }

    #[test]
    fn a_tone_lands_in_the_bin_that_holds_it() {
        let s = MelSettings::default();
        let mel = MelSpectrogram::new(s);
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let out = mel.compute(&samples);
        // Take the middle frame, away from the reflected edges.
        let t = 50;
        let loudest = (0..s.n_mels)
            .max_by(|&a, &b| out[a][t].partial_cmp(&out[b][t]).unwrap())
            .unwrap();
        let centre_hz = filter_centre_hz(&s, loudest);
        assert!(
            (centre_hz - 440.0).abs() < 60.0,
            "440 Hz landed in bin {loudest}, centred at {centre_hz} Hz"
        );
    }
}
