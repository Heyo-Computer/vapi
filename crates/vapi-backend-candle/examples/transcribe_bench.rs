//! vapi's Voxtral against the reference baseline.
//!
//! The number that decides whether the model is usable is not tokens per
//! second but the **realtime factor**: audio seconds transcribed per wall
//! second. One decoder position is 80 ms of audio, so a live session consumes
//! 12.5 steps a second and anything below 1.0x falls behind the speaker.
//!
//!     cargo run --release --features cuda --example transcribe_bench

use std::time::Instant;

use vapi_backend_candle::{CandleVoxtral, EncoderLoadOptions};
use vapi_core::config::{DType, DeviceKind};

const LENGTHS: [f32; 4] = [1.4, 10.0, 30.0, 60.0];
const SPEECH: &str = "/usr/share/sounds/alsa/Front_Center.wav";

/// Real speech, repeated to the requested length.
fn clip(seconds: f32) -> anyhow::Result<Vec<f32>> {
    let bytes = std::fs::read(SPEECH)?;
    // Minimal WAV reader: the one file this benchmark uses is 16-bit PCM.
    let at = bytes
        .windows(4)
        .position(|w| w == b"data")
        .ok_or_else(|| anyhow::anyhow!("no data chunk"))?
        + 8;
    let rate = u32::from_le_bytes(bytes[24..28].try_into()?) as usize;
    let channels = u16::from_le_bytes(bytes[22..24].try_into()?) as usize;
    let pcm: Vec<f32> = bytes[at..]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();
    let mono = vapi_audio::to_mono(&pcm, channels);
    let at_16k = vapi_audio::resample_linear(&mono, rate, 16_000);
    let want = (seconds * 16_000.0) as usize;
    Ok(at_16k.iter().cycle().take(want).copied().collect())
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/models/voxtral-realtime",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let cuda = cfg!(feature = "cuda");
    let started = Instant::now();
    let model = CandleVoxtral::load(
        &dir,
        &EncoderLoadOptions {
            dtype: if cuda { DType::Bf16 } else { DType::F32 },
            device: if cuda {
                DeviceKind::Cuda
            } else {
                DeviceKind::Cpu
            },
        },
    )?;
    let tokenizer = vapi_tokenize::Tokenization::from_dir(&dir)?;
    let tekken = tokenizer
        .tekken()
        .ok_or_else(|| anyhow::anyhow!("expected tekken.json"))?;
    let bos = tekken.special_id("<s>").unwrap();
    let pad = tekken.special_id("[STREAMING_PAD]").unwrap();
    println!("loaded in {:.1} s\n", started.elapsed().as_secs_f32());
    println!(
        "{:>8} {:>8} {:>10} {:>9}  transcript",
        "audio s", "wall s", "realtime", "steps/s"
    );

    for seconds in LENGTHS {
        let samples = clip(seconds)?;
        // One untimed pass first: the first call pays for CUDA context setup.
        let _ = model.transcribe(&samples, bos, pad)?;
        let t0 = Instant::now();
        let tokens = model.transcribe(&samples, bos, pad)?;
        let wall = t0.elapsed().as_secs_f32();
        let steps = tokens.len().saturating_sub(model.prompt_len());
        let text = tekken.decode(&tokens[model.prompt_len()..], true);
        let shown: String = text.trim().chars().take(40).collect();
        println!(
            "{seconds:>8.1} {wall:>8.2} {:>9.2}x {:>9.1}  {shown:?}",
            seconds / wall,
            steps as f32 / wall
        );
    }
    Ok(())
}
