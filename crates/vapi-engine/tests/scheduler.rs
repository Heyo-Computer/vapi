//! Scheduler behaviour, driven entirely by `MockBackend`.
//!
//! `MockBackend::forward` validates every batch it receives (slot collisions,
//! block-table coverage, position contiguity, cumulative-length consistency),
//! so each of these tests is also a block-accounting test.

use vapi_cache::{BlockPool, CacheNamespace};
use vapi_core::config::DType;
use vapi_core::{FinishReason, RequestId, SamplingParams};
use vapi_engine::backend::ExecutionBackend;
use vapi_engine::{MockBackend, Sampler, SamplerState, Scheduler, SchedulerConfig};

const BS: usize = 4;
const BLOCKS: usize = 64;

fn namespace() -> CacheNamespace {
    CacheNamespace::new("test", "fp", DType::F32, None, "global")
}

fn cfg() -> SchedulerConfig {
    SchedulerConfig {
        max_concurrent_seqs: 8,
        max_batched_tokens: 64,
        prefill_chunk_tokens: 8,
        max_context: 256,
        block_size: BS,
        prefix_cache: true,
    }
}

fn scheduler(cfg: SchedulerConfig) -> Scheduler {
    Scheduler::new(
        cfg.clone(),
        BlockPool::new(BLOCKS, BS, 0),
        namespace(),
        vec![0],
    )
}

/// Drive the engine until every admitted request finishes, returning the
/// generated tokens per request.
fn run(
    sched: &mut Scheduler,
    backend: &mut MockBackend,
    max_steps: usize,
) -> Vec<(RequestId, Vec<u32>, FinishReason)> {
    let mut out = Vec::new();
    let mut states: std::collections::HashMap<u64, SamplerState> = Default::default();

    for _ in 0..max_steps {
        let plan = sched.schedule();
        if !plan.is_empty() {
            let logits = backend
                .forward(&plan.batch)
                .expect("batch must be well-formed");
            let mut sampled = Vec::with_capacity(plan.sampled_seqs.len());
            for (row, seq_id) in plan.sampled_seqs.iter().enumerate() {
                let params = sched
                    .get(*seq_id)
                    .map(|s| s.params.clone())
                    .unwrap_or_default();
                let st = states
                    .entry(seq_id.0)
                    .or_insert_with(|| SamplerState::new(params.seed));
                let mut row_logits = logits.row(row).to_vec();
                sampled.push(Sampler::sample(&mut row_logits, &params, st));
            }
            sched.commit(&plan, &sampled);
        }

        for (_, seq) in sched.drain_finished() {
            let reason = match seq.status {
                vapi_engine::SeqStatus::Finished(r) => r,
                _ => FinishReason::Error,
            };
            out.push((seq.request_id.clone(), seq.generated().to_vec(), reason));
        }
        if sched.num_running() == 0 && sched.num_waiting() == 0 {
            break;
        }
    }
    out
}

#[test]
fn a_single_request_generates_its_scripted_tokens() {
    let mut s = scheduler(cfg());
    // Script ends in EOS (0), so generation stops on its own.
    let mut b = MockBackend::new(BLOCKS, BS).with_script([5, 6, 7, 0]);

    let rid = RequestId::new();
    s.admit(
        rid.clone(),
        vec![1, 2, 3],
        SamplingParams {
            temperature: 0.0,
            max_tokens: 20,
            ..Default::default()
        },
        "global".into(),
    )
    .unwrap();

    let done = run(&mut s, &mut b, 50);
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].1, vec![5, 6, 7, 0]);
    assert_eq!(done[0].2, FinishReason::Stop);
}

#[test]
fn max_tokens_is_respected_and_reported_as_length() {
    let mut s = scheduler(cfg());
    let mut b = MockBackend::new(BLOCKS, BS).with_script([9]); // never emits EOS

    s.admit(
        RequestId::new(),
        vec![1, 2],
        SamplingParams {
            temperature: 0.0,
            max_tokens: 3,
            ..Default::default()
        },
        "global".into(),
    )
    .unwrap();

    let done = run(&mut s, &mut b, 50);
    assert_eq!(done[0].1.len(), 3);
    assert_eq!(done[0].2, FinishReason::Length);
}

#[test]
fn every_block_is_returned_once_all_requests_finish() {
    // A leak here is the failure mode that wedges a server after days of
    // uptime, so assert the pool returns to pristine.
    let mut s = scheduler(cfg());
    let mut b = MockBackend::new(BLOCKS, BS).with_script([1, 2, 0]);

    for _ in 0..5 {
        s.admit(
            RequestId::new(),
            vec![1, 2, 3, 4, 5],
            SamplingParams {
                temperature: 0.0,
                max_tokens: 10,
                ..Default::default()
            },
            "global".into(),
        )
        .unwrap();
    }
    let done = run(&mut s, &mut b, 200);
    assert_eq!(done.len(), 5);

    s.pool().check_invariants();
    let stats = s.pool().stats();
    // Cached blocks stay resident but unreferenced; nothing may still be held.
    assert_eq!(
        stats.num_free, stats.num_blocks,
        "every block must be released"
    );
}

#[test]
fn a_long_prompt_is_prefilled_in_chunks_rather_than_one_forward() {
    let mut c = cfg();
    c.prefill_chunk_tokens = 4;
    c.max_batched_tokens = 4;
    let mut s = scheduler(c);
    let mut b = MockBackend::new(BLOCKS, BS).with_script([7, 0]);

    let prompt: Vec<u32> = (1..=16).collect();
    s.admit(
        RequestId::new(),
        prompt,
        SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        },
        "global".into(),
    )
    .unwrap();

    run(&mut s, &mut b, 100);

    let prefill_batches: Vec<usize> = b
        .batches_seen
        .iter()
        .map(|x| x.num_tokens())
        .take_while(|&n| n > 1)
        .collect();
    assert!(
        prefill_batches.len() >= 4,
        "16 tokens at chunk 4 should take several steps, got {prefill_batches:?}"
    );
    assert!(
        prefill_batches.iter().all(|&n| n <= 4),
        "no chunk may exceed the budget: {prefill_batches:?}"
    );
}

#[test]
fn only_the_final_prefill_chunk_produces_logits() {
    // Requesting logits for every position of a chunk would return
    // chunk_len x vocab floats to throw away.
    let mut c = cfg();
    c.prefill_chunk_tokens = 4;
    c.max_batched_tokens = 4;
    let mut s = scheduler(c);
    let mut b = MockBackend::new(BLOCKS, BS).with_script([7, 0]);

    s.admit(
        RequestId::new(),
        (1..=12).collect(),
        SamplingParams {
            temperature: 0.0,
            max_tokens: 2,
            ..Default::default()
        },
        "global".into(),
    )
    .unwrap();
    run(&mut s, &mut b, 100);

    let mid_chunks = &b.batches_seen[..2];
    for batch in mid_chunks {
        assert!(
            batch.logits_indices.is_empty(),
            "a mid-prefill chunk needs no logits"
        );
    }
}

#[test]
fn concurrent_requests_share_one_batch() {
    let mut s = scheduler(cfg());
    let mut b = MockBackend::new(BLOCKS, BS).with_script([3, 3, 0]);

    for _ in 0..4 {
        s.admit(
            RequestId::new(),
            vec![1, 2, 3, 4],
            SamplingParams {
                temperature: 0.0,
                max_tokens: 5,
                ..Default::default()
            },
            "global".into(),
        )
        .unwrap();
    }
    run(&mut s, &mut b, 100);

    let widest = b.batches_seen.iter().map(|x| x.batch_size()).max().unwrap();
    assert!(
        widest > 1,
        "continuous batching should coalesce requests, widest batch was {widest}"
    );
}

#[test]
fn a_shared_system_prompt_is_not_recomputed_for_the_second_user() {
    let mut s = scheduler(cfg());
    let mut b = MockBackend::new(BLOCKS, BS).with_script([4, 0]);

    let system: Vec<u32> = (100..108).collect(); // two full blocks

    let mut first = system.clone();
    first.extend([1, 2, 3, 4]);
    s.admit(
        RequestId::new(),
        first,
        SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        },
        "global".into(),
    )
    .unwrap();
    run(&mut s, &mut b, 100);

    let before = s.pool().stats();

    let mut second = system.clone();
    second.extend([9, 9, 9, 9]);
    s.admit(
        RequestId::new(),
        second,
        SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        },
        "global".into(),
    )
    .unwrap();
    run(&mut s, &mut b, 100);

    let after = s.pool().stats();
    assert!(
        after.hit_tokens > before.hit_tokens,
        "the second user should reuse the first user's KV blocks"
    );
    s.pool().check_invariants();
}

#[test]
fn prefix_cache_hits_do_not_change_the_output() {
    // The highest-value correctness test in the engine: a warm cache must be
    // a pure performance win. If this fails, slot_mapping or num_computed is
    // off by one and the server produces fluent, wrong text.
    let prompt: Vec<u32> = (1..=12).collect();
    let params = SamplingParams {
        temperature: 0.0,
        max_tokens: 6,
        ..Default::default()
    };

    let generate = |prefix_cache: bool, warm: bool| {
        let mut c = cfg();
        c.prefix_cache = prefix_cache;
        let mut s = scheduler(c);
        let mut b = MockBackend::new(BLOCKS, BS).with_script([11, 12, 13, 14, 15, 16]);
        if warm {
            s.admit(
                RequestId::new(),
                prompt.clone(),
                params.clone(),
                "global".into(),
            )
            .unwrap();
            run(&mut s, &mut b, 100);
        }
        s.admit(
            RequestId::new(),
            prompt.clone(),
            params.clone(),
            "global".into(),
        )
        .unwrap();
        let done = run(&mut s, &mut b, 100);
        done.last().expect("a completion").1.clone()
    };

    let cold = generate(false, false);
    let warm = generate(true, true);
    assert_eq!(cold, warm, "a cache hit must not alter generated tokens");
}

#[test]
fn a_cache_hit_covering_the_whole_prompt_still_generates() {
    // Pathological case: every prompt block is cached, so there is nothing
    // left to forward and therefore no logits to sample from. The scheduler
    // must give back the last block and recompute it.
    let prompt: Vec<u32> = (1..=8).collect(); // exactly two blocks
    let params = SamplingParams {
        temperature: 0.0,
        max_tokens: 3,
        ..Default::default()
    };

    let mut s = scheduler(cfg());
    let mut b = MockBackend::new(BLOCKS, BS).with_script([21, 22, 23]);

    s.admit(
        RequestId::new(),
        prompt.clone(),
        params.clone(),
        "global".into(),
    )
    .unwrap();
    let first = run(&mut s, &mut b, 100);

    s.admit(
        RequestId::new(),
        prompt.clone(),
        params.clone(),
        "global".into(),
    )
    .unwrap();
    let second = run(&mut s, &mut b, 100);

    assert_eq!(second.len(), 1, "a fully cached prompt must still complete");
    assert!(!second[0].1.is_empty(), "it must still generate tokens");
    assert_eq!(first[0].1, second[0].1);
}

#[test]
fn cancelling_a_request_frees_its_blocks_immediately() {
    let mut s = scheduler(cfg());
    let mut b = MockBackend::new(BLOCKS, BS).with_script([1]); // never stops

    let rid = RequestId::new();
    s.admit(
        rid.clone(),
        vec![1, 2, 3, 4, 5, 6],
        SamplingParams {
            temperature: 0.0,
            max_tokens: 1000,
            ..Default::default()
        },
        "global".into(),
    )
    .unwrap();

    // Run a few steps so it is genuinely mid-flight holding blocks.
    for _ in 0..3 {
        let plan = s.schedule();
        if plan.is_empty() {
            break;
        }
        let logits = b.forward(&plan.batch).unwrap();
        let sampled: Vec<u32> = (0..plan.sampled_seqs.len())
            .map(|r| {
                let mut l = logits.row(r).to_vec();
                Sampler::sample(
                    &mut l,
                    &SamplingParams {
                        temperature: 0.0,
                        ..Default::default()
                    },
                    &mut SamplerState::new(None),
                )
            })
            .collect();
        s.commit(&plan, &sampled);
    }
    assert!(
        s.pool().stats().num_free < BLOCKS,
        "should be holding blocks"
    );

    s.cancel(&rid).expect("the request is live");
    let done = s.drain_finished();
    assert_eq!(done.len(), 1);
    assert_eq!(
        done[0].1.status,
        vapi_engine::SeqStatus::Finished(FinishReason::Cancelled)
    );

    s.pool().check_invariants();
    assert_eq!(
        s.pool().stats().num_free,
        BLOCKS,
        "cancellation must release every block"
    );
    assert_eq!(s.num_running(), 0);
}

#[test]
fn an_over_long_prompt_is_rejected_at_admission() {
    let mut c = cfg();
    c.max_context = 8;
    let mut s = scheduler(c);
    let err = s.admit(
        RequestId::new(),
        (1..=32).collect(),
        SamplingParams::default(),
        "global".into(),
    );
    assert!(matches!(
        err,
        Err(vapi_core::Error::ContextLengthExceeded { .. })
    ));
}

#[test]
fn admission_headroom_reflects_load() {
    let mut s = scheduler(cfg()); // max_concurrent_seqs = 8
    assert_eq!(s.admission_headroom(), 8);
    for _ in 0..3 {
        s.admit(
            RequestId::new(),
            vec![1, 2],
            SamplingParams::default(),
            "global".into(),
        )
        .unwrap();
    }
    assert_eq!(s.admission_headroom(), 5, "this is the NATS pull credit");
}

#[test]
fn running_out_of_blocks_preempts_rather_than_failing() {
    // Tiny pool, oversubscribed. Every request must still complete correctly.
    let mut c = cfg();
    c.max_concurrent_seqs = 8;
    c.prefill_chunk_tokens = 4;
    let mut s = Scheduler::new(c, BlockPool::new(8, BS, 0), namespace(), vec![0]);
    let mut b = MockBackend::new(8, BS).with_script([5, 6, 0]);

    for _ in 0..6 {
        s.admit(
            RequestId::new(),
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            SamplingParams {
                temperature: 0.0,
                max_tokens: 4,
                ..Default::default()
            },
            "global".into(),
        )
        .unwrap();
    }
    let done = run(&mut s, &mut b, 2000);

    assert_eq!(
        done.len(),
        6,
        "every request must complete despite the pressure"
    );
    s.pool().check_invariants();
    assert_eq!(
        s.pool().stats().num_free,
        8,
        "no blocks leaked under preemption"
    );
}
