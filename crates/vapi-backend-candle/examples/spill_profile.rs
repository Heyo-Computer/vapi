//! Is restoring a KV block cheaper than recomputing it?
//!
//! Tier 3 only pays off when copying a block back from host memory beats
//! prefilling the `BLOCK_SIZE` tokens it holds. Both are measured here on
//! the real model, at batch 1 and at a batch size a busy server actually
//! prefills with.
//!
//! ```text
//! cargo run --release --features cuda --target-dir target/cuda \
//!   --example spill_profile -- ~/models/lfm2.5-2.6b
//! ```

use std::time::Instant;

use vapi_backend_candle::{CandleBackend, LoadOptions};
use vapi_core::config::BLOCK_SIZE;
use vapi_engine::backend::{ExecutionBackend, ForwardBatch};

fn prefill_batch(seqs: usize, blocks_per_seq: usize) -> ForwardBatch {
    let per_seq = blocks_per_seq * BLOCK_SIZE;
    let mut b = ForwardBatch {
        max_seqlen_q: per_seq,
        max_seqlen_k: per_seq,
        ..Default::default()
    };
    b.cu_seqlens_q.push(0);
    b.cu_seqlens_k.push(0);
    for s in 0..seqs {
        let table: Vec<u32> = (0..blocks_per_seq)
            .map(|i| (s * blocks_per_seq + i) as u32)
            .collect();
        for pos in 0..per_seq {
            b.tokens
                .push(((s * 7919 + pos * 104729) % 30000 + 100) as u32);
            b.positions.push(pos as u32);
            b.slot_mapping
                .push(table[pos / BLOCK_SIZE] * BLOCK_SIZE as u32 + (pos % BLOCK_SIZE) as u32);
        }
        b.block_tables.push(table);
        b.cu_seqlens_q.push(((s + 1) * per_seq) as u32);
        b.cu_seqlens_k.push(((s + 1) * per_seq) as u32);
        b.logits_indices.push(((s + 1) * per_seq - 1) as u32);
    }
    b
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: spill_profile <model-dir>");
    let blocks_per_seq = 8;
    let max_seqs = 16;
    let opts = LoadOptions {
        num_blocks: Some(max_seqs * blocks_per_seq + 4),
        ..LoadOptions::default()
    };
    let mut backend = CandleBackend::load(&dir, &opts)?;
    let bytes = backend.block_bytes();
    println!(
        "block = {BLOCK_SIZE} tokens = {:.2} MB of KV",
        bytes as f64 / 1e6
    );

    // Warm up the allocator and any lazily loaded kernels.
    let _ = backend.forward(&prefill_batch(1, 1))?;

    let n = 64;
    let mut blobs = Vec::with_capacity(n);
    let t = Instant::now();
    for i in 0..n {
        blobs.push(backend.export_block(vapi_cache::BlockId(i as u32))?);
    }
    let export_ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;

    let t = Instant::now();
    for (i, b) in blobs.iter().enumerate() {
        backend.import_block(vapi_cache::BlockId(i as u32), b)?;
    }
    let import_ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;

    println!(
        "export {export_ms:.3} ms/block ({:.1} GB/s), import {import_ms:.3} ms/block ({:.1} GB/s)",
        bytes as f64 / export_ms / 1e6,
        bytes as f64 / import_ms / 1e6
    );

    println!("\nrecompute cost of the same block, by prefill batch:");
    println!(
        "{:>5} {:>10} {:>14} {:>16}",
        "seqs", "tokens", "prefill ms", "ms per block"
    );
    for seqs in [1usize, 4, 16] {
        let batch = prefill_batch(seqs, blocks_per_seq);
        let tokens = batch.tokens.len();
        for _ in 0..2 {
            let _ = backend.forward(&batch)?;
        }
        let reps = 5;
        let t = Instant::now();
        for _ in 0..reps {
            let _ = backend.forward(&batch)?;
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let per_block = ms / (seqs * blocks_per_seq) as f64;
        println!("{seqs:>5} {tokens:>10} {ms:>14.2} {per_block:>16.3}");
        if seqs == 1 {
            println!(
                "      restore is {:.1}x the cost of recompute at this batch",
                import_ms / per_block
            );
        }
    }
    Ok(())
}
