//! A minimal WAV reader.
//!
//! WAV only, and deliberately: decoding MP3 or AAC means a codec dependency
//! and a much larger attack surface for something a caller controls. A
//! gateway that says "send me WAV" is more honest than one that accepts
//! anything and fails deep inside a decoder.

use vapi_core::{Error, Result};

/// Decoded PCM, as the frontend wants it.
#[derive(Clone, Debug)]
pub struct Pcm {
    /// Interleaved samples in `-1.0..=1.0`.
    pub samples: Vec<f32>,
    pub sample_rate: usize,
    pub channels: usize,
}

impl Pcm {
    pub fn seconds(&self) -> f32 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.samples.len() as f32 / (self.sample_rate * self.channels) as f32
    }

    /// Mono at `rate`, which is what a model's frontend expects.
    pub fn into_mono(self, rate: usize) -> Vec<f32> {
        let mono = crate::to_mono(&self.samples, self.channels);
        crate::resample_linear(&mono, self.sample_rate, rate)
    }
}

fn u16_at(b: &[u8], at: usize) -> Result<u16> {
    b.get(at..at + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| Error::InvalidRequest("wav: truncated header".into()))
}

fn u32_at(b: &[u8], at: usize) -> Result<u32> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| Error::InvalidRequest("wav: truncated header".into()))
}

/// Parse a RIFF/WAVE file: PCM or IEEE float, 8/16/24/32-bit.
///
/// Chunks are walked rather than assumed to be in a fixed order — plenty of
/// encoders put a `LIST` before the `data`, and a reader that assumes byte 44
/// works until it meets one.
pub fn decode(bytes: &[u8]) -> Result<Pcm> {
    let bad = |m: &str| Error::InvalidRequest(format!("wav: {m}"));
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }

    let mut at = 12;
    let mut format: Option<(u16, u16, usize)> = None; // (tag, bits, channels)
    let mut rate = 0usize;
    let mut data: Option<&[u8]> = None;

    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = u32_at(bytes, at + 4)? as usize;
        let body = at + 8;
        let end = body.saturating_add(size).min(bytes.len());
        match id {
            b"fmt " => {
                let tag = u16_at(bytes, body)?;
                let channels = u16_at(bytes, body + 2)? as usize;
                rate = u32_at(bytes, body + 4)? as usize;
                let bits = u16_at(bytes, body + 14)?;
                format = Some((tag, bits, channels));
            }
            b"data" => data = Some(&bytes[body..end]),
            _ => {}
        }
        // Chunks are word-aligned; an odd size carries a pad byte.
        at = body + size + (size & 1);
    }

    let (tag, bits, channels) = format.ok_or_else(|| bad("no fmt chunk"))?;
    let data = data.ok_or_else(|| bad("no data chunk"))?;
    if channels == 0 || rate == 0 {
        return Err(bad("zero channels or sample rate"));
    }
    // 1 = PCM, 3 = IEEE float, 0xFFFE = extensible (whose real tag is in the
    // extension; both of the ones we support look the same here).
    let samples = match (tag, bits) {
        (1 | 0xFFFE, 8) => data.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect(),
        (1 | 0xFFFE, 16) => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
        (1 | 0xFFFE, 24) => data
            .chunks_exact(3)
            .map(|c| {
                // Sign-extend the 24-bit value into an i32.
                let v = ((c[2] as i32) << 24 | (c[1] as i32) << 16 | (c[0] as i32) << 8) >> 8;
                v as f32 / 8_388_608.0
            })
            .collect(),
        (1 | 0xFFFE, 32) => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2_147_483_648.0)
            .collect(),
        (3, 32) => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        (t, b) => {
            return Err(bad(&format!(
                "unsupported format tag {t} at {b} bits; PCM 8/16/24/32 and float32 are supported"
            )));
        }
    };

    Ok(Pcm {
        samples,
        sample_rate: rate,
        channels,
    })
}

/// 16-bit little-endian PCM, for putting audio on the wire.
pub fn to_i16_le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// The inverse of [`to_i16_le`].
pub fn from_i16_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(
        tag: u16,
        bits: u16,
        channels: u16,
        rate: u32,
        body: &[u8],
        extra_chunk: bool,
    ) -> Vec<u8> {
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&tag.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&0u32.to_le_bytes()); // byte rate, unread
        fmt.extend_from_slice(&0u16.to_le_bytes()); // block align, unread
        fmt.extend_from_slice(&bits.to_le_bytes());

        let mut chunks = Vec::new();
        chunks.extend_from_slice(b"fmt ");
        chunks.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        chunks.extend_from_slice(&fmt);
        if extra_chunk {
            // A LIST chunk of odd length, to exercise both the walk and the
            // pad byte.
            chunks.extend_from_slice(b"LIST");
            chunks.extend_from_slice(&5u32.to_le_bytes());
            chunks.extend_from_slice(b"INFOx");
            chunks.push(0);
        }
        chunks.extend_from_slice(b"data");
        chunks.extend_from_slice(&(body.len() as u32).to_le_bytes());
        chunks.extend_from_slice(body);

        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((chunks.len() + 4) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(&chunks);
        out
    }

    #[test]
    fn sixteen_bit_mono_round_trips() {
        let body = to_i16_le(&[0.0, 0.5, -0.5, 1.0]);
        let pcm = decode(&wav(1, 16, 1, 16_000, &body, false)).unwrap();
        assert_eq!(pcm.sample_rate, 16_000);
        assert_eq!(pcm.channels, 1);
        assert_eq!(pcm.samples.len(), 4);
        assert!((pcm.samples[1] - 0.5).abs() < 1e-3, "{:?}", pcm.samples);
        assert!((pcm.samples[2] + 0.5).abs() < 1e-3);
    }

    #[test]
    fn a_chunk_before_the_data_does_not_derail_the_reader() {
        // The failure this prevents: assuming the data starts at byte 44,
        // which works on every file an encoder wrote without metadata.
        let body = to_i16_le(&[0.25; 8]);
        let pcm = decode(&wav(1, 16, 1, 8_000, &body, true)).unwrap();
        assert_eq!(pcm.samples.len(), 8);
        assert!((pcm.samples[0] - 0.25).abs() < 1e-3);
    }

    #[test]
    fn float_and_twenty_four_bit_are_understood() {
        let floats: Vec<u8> = [0.5f32, -0.25]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let pcm = decode(&wav(3, 32, 1, 44_100, &floats, false)).unwrap();
        assert!((pcm.samples[0] - 0.5).abs() < 1e-6);
        assert!((pcm.samples[1] + 0.25).abs() < 1e-6);

        // 24-bit: half scale is 0x400000.
        let body = vec![0x00, 0x00, 0x40, 0x00, 0x00, 0xC0];
        let pcm = decode(&wav(1, 24, 1, 48_000, &body, false)).unwrap();
        assert!((pcm.samples[0] - 0.5).abs() < 1e-5, "{:?}", pcm.samples);
        assert!((pcm.samples[1] + 0.5).abs() < 1e-5, "{:?}", pcm.samples);
    }

    #[test]
    fn stereo_becomes_mono_at_the_rate_the_model_wants() {
        let body = to_i16_le(&[1.0, 0.0, 1.0, 0.0]); // L R L R
        let pcm = decode(&wav(1, 16, 2, 32_000, &body, false)).unwrap();
        assert_eq!(pcm.channels, 2);
        assert!((pcm.seconds() - 2.0 / 32_000.0).abs() < 1e-6);
        let mono = pcm.into_mono(16_000);
        // Two stereo frames averaged to 0.5, then halved in rate to one.
        assert_eq!(mono.len(), 1);
        assert!((mono[0] - 0.5).abs() < 1e-3, "{mono:?}");
    }

    #[test]
    fn something_that_is_not_a_wav_says_so() {
        let err = decode(b"ID3\x04not an mp3 either").unwrap_err().to_string();
        assert!(err.contains("RIFF/WAVE"), "{err}");
    }

    #[test]
    fn an_unsupported_bit_depth_names_itself() {
        let err = decode(&wav(1, 12, 1, 16_000, &[0; 4], false))
            .unwrap_err()
            .to_string();
        assert!(err.contains("12 bits"), "{err}");
    }

    #[test]
    fn i16_survives_the_round_trip_to_the_wire() {
        let samples = [0.0, 0.5, -0.5, 0.999, -0.999];
        let back = from_i16_le(&to_i16_le(&samples));
        assert_eq!(back.len(), samples.len());
        for (a, b) in samples.iter().zip(&back) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}
