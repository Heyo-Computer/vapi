//! Drive batch-64 decode steps directly against the backend, for profiling.
//!
//! ```text
//! cargo run --release --features cuda --target-dir target/cuda \
//!   --example decode_profile -- ~/models/lfm2.5-2.6b [batch] [steps]
//! VAPI_CUDA_GRAPHS=1 nsys profile --cuda-graph-trace=node --stats=true ...
//! ```
//!
//! Prefills `batch` random 32-token prompts in one forward, then runs `steps`
//! greedy decode steps and reports the per-step time. With
//! `VAPI_CUDA_GRAPHS=1` the decode steps go through graph replay.

use std::time::Instant;

use vapi_backend_candle::{CandleBackend, LoadOptions};
use vapi_core::config::BLOCK_SIZE;
use vapi_engine::backend::{ExecutionBackend, ForwardBatch};

const PROMPT: usize = 32;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .expect("usage: decode_profile <model-dir> [batch] [steps]");
    let batch: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(64);
    let steps: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(20);
    let cuda_graphs = std::env::var("VAPI_CUDA_GRAPHS").is_ok_and(|v| v == "1");

    let blocks_per_seq = (PROMPT + steps).div_ceil(BLOCK_SIZE);
    let num_blocks = batch * blocks_per_seq + 4;
    let opts = LoadOptions {
        num_blocks: Some(num_blocks),
        cuda_graphs,
        ..LoadOptions::default()
    };
    let t = Instant::now();
    let mut backend = CandleBackend::load(&dir, &opts)?;
    eprintln!(
        "loaded in {:.1}s, cuda_graphs={cuda_graphs}",
        t.elapsed().as_secs_f32()
    );

    let block_tables: Vec<Vec<u32>> = (0..batch)
        .map(|i| {
            (0..blocks_per_seq)
                .map(|b| (i * blocks_per_seq + b) as u32)
                .collect()
        })
        .collect();
    let slot = |seq: usize, pos: usize| {
        block_tables[seq][pos / BLOCK_SIZE] * BLOCK_SIZE as u32 + (pos % BLOCK_SIZE) as u32
    };

    // Prefill: deterministic pseudo-random prompt ids.
    let mut prefill = ForwardBatch {
        block_tables: block_tables.clone(),
        max_seqlen_q: PROMPT,
        max_seqlen_k: PROMPT,
        ..Default::default()
    };
    prefill.cu_seqlens_q.push(0);
    prefill.cu_seqlens_k.push(0);
    for seq in 0..batch {
        for pos in 0..PROMPT {
            prefill
                .tokens
                .push(((seq * 7919 + pos * 104729) % 30000 + 100) as u32);
            prefill.positions.push(pos as u32);
            prefill.slot_mapping.push(slot(seq, pos));
        }
        prefill.cu_seqlens_q.push(((seq + 1) * PROMPT) as u32);
        prefill.cu_seqlens_k.push(((seq + 1) * PROMPT) as u32);
        prefill.logits_indices.push(((seq + 1) * PROMPT - 1) as u32);
    }
    let t = Instant::now();
    let logits = backend.forward(&prefill)?;
    let mut next: Vec<u32> = (0..batch).map(|i| argmax(logits.row(i))).collect();
    eprintln!(
        "prefill {} tokens: {:.1} ms",
        batch * PROMPT,
        t.elapsed().as_secs_f32() * 1e3
    );

    let mut times = Vec::with_capacity(steps);
    for step in 0..steps {
        let pos = PROMPT + step;
        let b = ForwardBatch {
            tokens: next.clone(),
            positions: vec![pos as u32; batch],
            cu_seqlens_q: (0..=batch as u32).collect(),
            cu_seqlens_k: (0..=batch).map(|i| (i * (pos + 1)) as u32).collect(),
            slot_mapping: (0..batch).map(|s| slot(s, pos)).collect(),
            block_tables: block_tables.clone(),
            logits_indices: (0..batch as u32).collect(),
            max_seqlen_q: 1,
            max_seqlen_k: pos + 1,
        };
        b.validate(num_blocks, BLOCK_SIZE)?;
        let t = Instant::now();
        next = match backend.forward_greedy(&b)? {
            Some(ids) => ids,
            None => {
                let logits = backend.forward(&b)?;
                (0..batch).map(|i| argmax(logits.row(i))).collect()
            }
        };
        times.push(t.elapsed().as_secs_f32() * 1e3);
    }
    let warm = &times[times.len().min(3)..];
    let mean = warm.iter().sum::<f32>() / warm.len().max(1) as f32;
    let mut sorted = warm.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "decode batch={batch}: first {:.1} ms, then mean {:.2} ms, min {:.2}, max {:.2} ({} steps)",
        times[0],
        mean,
        sorted.first().copied().unwrap_or(0.0),
        sorted.last().copied().unwrap_or(0.0),
        warm.len()
    );
    eprintln!("last ids: {:?}", &next[..next.len().min(8)]);
    Ok(())
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}
