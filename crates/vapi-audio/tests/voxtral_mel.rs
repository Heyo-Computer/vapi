//! The frontend against the reference feature extractor.
//!
//! Every later check depends on this one: a mel bin that is slightly wrong
//! feeds an encoder that is slightly wrong, and the transcript is merely
//! worse rather than broken, which is the hardest kind of bug to find later.

use std::path::Path;

use vapi_audio::{MelSettings, MelSpectrogram};

fn golden() -> Option<serde_json::Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/goldens/voxtral-mel.json");
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

#[test]
fn log_mel_frames_match_the_reference() {
    let Some(g) = golden() else {
        eprintln!("skipping: tests/goldens/voxtral-mel.json not generated");
        return;
    };
    let s = &g["settings"];
    let settings = MelSettings {
        sampling_rate: s["sampling_rate"].as_u64().unwrap() as usize,
        n_fft: s["n_fft"].as_u64().unwrap() as usize,
        win_length: s["win_length"].as_u64().unwrap() as usize,
        hop_length: s["hop_length"].as_u64().unwrap() as usize,
        n_mels: s["feature_size"].as_u64().unwrap() as usize,
        max_frequency: s["sampling_rate"].as_u64().unwrap() as f32 / 2.0,
        global_log_mel_max: s["global_log_mel_max"].as_f64().unwrap() as f32,
    };
    let samples: Vec<f32> = serde_json::from_value(g["samples"].clone()).unwrap();
    let want: Vec<Vec<f32>> = serde_json::from_value(g["mel"].clone()).unwrap();

    let got = MelSpectrogram::new(settings).compute(&samples);
    assert_eq!(got.len(), want.len(), "mel bins");
    assert_eq!(got[0].len(), want[0].len(), "frames");

    let mut worst = 0f32;
    let mut worst_at = (0, 0);
    for (m, (a, b)) in got.iter().zip(&want).enumerate() {
        for (t, (x, y)) in a.iter().zip(b).enumerate() {
            let d = (x - y).abs();
            if d > worst {
                worst = d;
                worst_at = (m, t);
            }
        }
    }
    let (m, t) = worst_at;
    assert!(
        worst < 2e-4,
        "worst drift {worst} at mel {m} frame {t}: {} vs {}",
        got[m][t],
        want[m][t]
    );
    eprintln!(
        "mel: {} bins x {} frames, worst drift {worst:.2e}",
        got.len(),
        got[0].len()
    );
}
