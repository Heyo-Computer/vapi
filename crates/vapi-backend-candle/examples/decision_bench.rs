//! Latency of one decision pass, against the published reference numbers.
//!
//! A pass costs `rows × longest row` positions of work, so both are swept.
//! The reference figures are a Tesla T4: 39.5 ms for one question and 158.6 ms
//! for ten.
//!
//!     cargo run --release --features cuda --example decision_bench -- ~/models/laya

use std::time::Instant;

use vapi_backend_candle::{CandleEncoder, EncoderLoadOptions};
use vapi_core::QuestionType;
use vapi_core::config::{DType, DeviceKind};
use vapi_engine::{EncoderBackend, EncoderBatch, EncoderRow};

const ROWS: [usize; 5] = [1, 4, 8, 16, 64];
const LENGTHS: [usize; 3] = [96, 256, 512];
const WARMUP: usize = 3;
const ITERS: usize = 20;

fn row(len: usize, seed: u32) -> EncoderRow {
    EncoderRow {
        // Content does not change the cost; shape does.
        tokens: (0..len)
            .map(|i| (i as u32 * 7 + seed) % 50_000 + 8)
            .collect(),
        markers: vec![2, 6, 10, 14],
        qtype: QuestionType::Choice,
    }
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("{}/models/laya", std::env::var("HOME").unwrap_or_default()));
    let cuda = !cfg!(not(feature = "cuda"));
    let mut encoder = CandleEncoder::load(
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
    println!("{dir}\n");
    println!(
        "{:>6} {:>8} {:>10} {:>10} {:>12}",
        "rows", "tokens", "ms", "ms/row", "rows/s"
    );

    for len in LENGTHS {
        for rows in ROWS {
            let batch = EncoderBatch::new((0..rows).map(|i| row(len, i as u32)).collect());
            for _ in 0..WARMUP {
                encoder.forward(&batch)?;
            }
            let start = Instant::now();
            for _ in 0..ITERS {
                encoder.forward(&batch)?;
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
            println!(
                "{rows:>6} {len:>8} {ms:>10.1} {:>10.2} {:>12.0}",
                ms / rows as f64,
                rows as f64 * 1000.0 / ms
            );
        }
        println!();
    }
    Ok(())
}
