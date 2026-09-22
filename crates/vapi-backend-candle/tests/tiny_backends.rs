//! `CandleBackend` on the tiny fixtures, driven by the real scheduler:
//! dispatch on `model_type`, the spec, and greedy generation against the
//! reference implementation's own `generate()`. Skips without them.

#![cfg(feature = "candle")]

use std::path::{Path, PathBuf};

use vapi_backend_candle::{CandleBackend, LoadOptions};
use vapi_cache::{BlockPool, CacheNamespace};
use vapi_core::config::{BLOCK_SIZE, DType, DeviceKind};
use vapi_core::{RequestId, SamplingParams};
use vapi_engine::backend::ExecutionBackend;
use vapi_engine::{Scheduler, SchedulerConfig};

#[derive(serde::Deserialize)]
struct Golden {
    prompt_ids: Vec<u32>,
    greedy_ids: Vec<u32>,
}

fn fixture(name: &str) -> Option<(PathBuf, Golden)> {
    let home = std::env::var("HOME").unwrap_or_default();
    let dir = Path::new(&home).join("models").join(name);
    if !dir.join("config.json").exists() {
        eprintln!("SKIP: no fixture at {}", dir.display());
        return None;
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/goldens")
        .join(format!("{name}.json"));
    Some((
        dir,
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?,
    ))
}

fn greedy_through_the_scheduler(dir: &Path, prompt: &[u32], n: usize) -> (Vec<u32>, CandleBackend) {
    let opts = LoadOptions {
        dtype: DType::F32,
        device: DeviceKind::Cpu,
        num_blocks: Some(16),
        kv_cache_fraction: 0.85,
        cuda_graphs: false,
    };
    greedy_with(dir, prompt, n, &opts)
}

fn greedy_with(
    dir: &Path,
    prompt: &[u32],
    n: usize,
    opts: &LoadOptions,
) -> (Vec<u32>, CandleBackend) {
    let mut backend =
        CandleBackend::load(dir, opts).expect("the model loads through the generic backend");
    let cfg = SchedulerConfig {
        max_concurrent_seqs: 4,
        max_batched_tokens: 64,
        prefill_chunk_tokens: 7, // several chunks for an 18-token prompt
        max_context: 128,
        block_size: BLOCK_SIZE,
        prefix_cache: true,
    };
    let mut sched = Scheduler::new(
        cfg,
        BlockPool::new(16, BLOCK_SIZE, 0),
        CacheNamespace::new("t", "fp", DType::F32, None, "global"),
        vec![], // the fixtures have no EOS; run to max_tokens
    );
    let rid = RequestId::new();
    sched
        .admit(
            rid,
            prompt.to_vec(),
            SamplingParams {
                temperature: 0.0,
                max_tokens: n,
                ..Default::default()
            },
            "global".into(),
        )
        .unwrap();
    let mut out = Vec::new();
    for _ in 0..1000 {
        let plan = sched.schedule();
        if !plan.is_empty() {
            let logits = backend.forward(&plan.batch).unwrap();
            let sampled: Vec<u32> = (0..plan.sampled_seqs.len())
                .map(|r| {
                    let row = logits.row(r);
                    (0..row.len())
                        .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                        .unwrap() as u32
                })
                .collect();
            sched.commit(&plan, &sampled);
        }
        for (_, seq) in sched.drain_finished() {
            out = seq.generated().to_vec();
        }
        if sched.num_running() == 0 && sched.num_waiting() == 0 {
            break;
        }
    }
    (out, backend)
}

fn check(name: &str) {
    let Some((dir, g)) = fixture(name) else {
        return;
    };
    let (got, backend) = greedy_through_the_scheduler(&dir, &g.prompt_ids, g.greedy_ids.len());
    assert_eq!(
        got, g.greedy_ids,
        "{name}: greedy ids vs the reference generate()"
    );
    let spec = backend.spec();
    assert_eq!(spec.vocab_size, 256);
    assert_eq!(spec.num_kv_heads, 2);
    assert_eq!(spec.head_dim, 16);
    eprintln!("{name}: {} greedy ids match the reference", got.len());
}

#[test]
fn the_backend_dispatches_on_model_type_and_reproduces_reference_greedy_ids() {
    check("laguna-tiny");
}

#[test]
fn the_xs_feature_set_reproduces_reference_greedy_ids_through_the_scheduler() {
    check("laguna-tiny-xs");
}

#[test]
fn lfm2_reproduces_transformers_greedy_ids_through_the_scheduler() {
    // The conv layers' per-token rows and the attention layers' K/V share
    // one paged cache with per-layer layouts; chunked prefill crosses the
    // conv window several times.
    check("lfm2-tiny");
}

/// The captured decode graph must produce exactly what eager execution
/// does, step for step, including when prefill chunks (eager) and decodes
/// (graph) interleave and the batch size crosses bucket boundaries.
#[cfg(feature = "cuda")]
#[test]
fn cuda_graphs_reproduce_eager_greedy_ids() {
    let Some((dir, g)) = fixture("lfm2-tiny") else {
        return;
    };
    let base = LoadOptions {
        dtype: DType::Bf16,
        device: DeviceKind::Cuda,
        num_blocks: Some(16),
        kv_cache_fraction: 0.85,
        cuda_graphs: false,
    };
    let (eager, _) = greedy_with(&dir, &g.prompt_ids, 32, &base);
    let (graphed, backend) = greedy_with(
        &dir,
        &g.prompt_ids,
        32,
        &LoadOptions {
            cuda_graphs: true,
            ..base
        },
    );
    assert_eq!(graphed, eager, "graph replay vs eager on the GPU");
    assert_eq!(
        backend.num_blocks(),
        14,
        "two blocks reserved for padding rows"
    );
}
