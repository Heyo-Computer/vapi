//! Golden-token tests against a real Llama-architecture model.
//!
//! These need weights on disk, so they skip (loudly) when none are found.
//! Point `VAPI_TEST_MODEL_DIR` at a local model directory, or put one at
//! `~/models/smollm2-135m-instruct` (the default the goldens in
//! `tests/goldens/` were generated from). Regenerate goldens with
//! `tools/gen_goldens.py`.
//!
//! On the CPU in f32 the greedy ids must match Hugging Face exactly. On a GPU
//! in bf16 they are expected to diverge eventually at a near-tie; the test
//! then reports the first divergent index and the logit gap there, and only
//! fails when the gap is large enough that it cannot be rounding.

#![cfg(feature = "candle")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use vapi_backend_candle::{CandleBackend, LoadOptions};
use vapi_cache::{BlockPool, CacheNamespace};
use vapi_core::config::{BLOCK_SIZE, DType, DeviceKind};
use vapi_core::{FinishReason, RequestId, SamplingParams, SeqId};
use vapi_engine::backend::ExecutionBackend;
use vapi_engine::{Scheduler, SchedulerConfig, SeqStatus};
use vapi_openai::{ChatMessage, ChatRole};
use vapi_tokenize::TokenizerBundle;

const NUM_BLOCKS: usize = 96;

#[derive(serde::Deserialize)]
struct Golden {
    model_id: String,
    fixture: String,
    config: serde_json::Value,
    messages: Vec<GoldenMessage>,
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    rendered: String,
    prompt_ids: Vec<u32>,
    greedy_ids: Vec<u32>,
    greedy_text: String,
    max_new_tokens: usize,
}

#[derive(serde::Deserialize)]
struct GoldenMessage {
    role: String,
    content: String,
    #[serde(default)]
    tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

impl GoldenMessage {
    fn to_chat(&self) -> ChatMessage {
        let role = match self.role.as_str() {
            "system" => ChatRole::System,
            "assistant" => ChatRole::Assistant,
            "tool" => ChatRole::Tool,
            _ => ChatRole::User,
        };
        let mut m = ChatMessage::user(self.content.clone());
        m.role = role;
        m.tool_call_id = self.tool_call_id.clone();
        // The fixture writes arguments as a mapping (what transformers
        // takes); the API takes them JSON-encoded, as OpenAI sends them.
        m.tool_calls = self.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|c| {
                    vapi_openai::ToolCall::function(
                        c["id"].as_str().unwrap_or("call_0"),
                        c["function"]["name"].as_str().unwrap_or(""),
                        c["function"]["arguments"].to_string(),
                    )
                })
                .collect()
        });
        m
    }
}

fn model_dir() -> Option<PathBuf> {
    let dir = match std::env::var("VAPI_TEST_MODEL_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home).join("models/smollm2-135m-instruct")
        }
    };
    if dir.join("config.json").exists() && dir.join("tokenizer.json").exists() {
        Some(dir)
    } else {
        eprintln!(
            "SKIP: no model at {}; set VAPI_TEST_MODEL_DIR to run the real-model tests",
            dir.display()
        );
        None
    }
}

/// Goldens generated from the model in `dir`, matched on its config.
fn goldens_for(dir: &Path) -> Vec<Golden> {
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/goldens");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&root).expect("tests/goldens exists") {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        // Other golden families (Laguna's, say) share the directory; skip
        // anything that is not a chat golden.
        let Ok(g) = serde_json::from_str::<Golden>(&std::fs::read_to_string(&path).unwrap()) else {
            continue;
        };
        let same_model = g
            .config
            .as_object()
            .unwrap()
            .iter()
            .all(|(k, v)| config.get(k) == Some(v));
        if same_model {
            out.push(g);
        }
    }
    assert!(
        !out.is_empty(),
        "no goldens in {} match the model at {}; run tools/gen_goldens.py",
        root.display(),
        dir.display()
    );
    out.sort_by(|a, b| a.fixture.cmp(&b.fixture));
    out
}

fn load(dir: &Path) -> CandleBackend {
    load_with(dir, NUM_BLOCKS)
}

fn load_with(dir: &Path, num_blocks: usize) -> CandleBackend {
    let opts = LoadOptions {
        dtype: DType::Auto,
        device: DeviceKind::Auto,
        num_blocks: Some(num_blocks),
        kv_cache_fraction: 0.85,
        // VAPI_TEST_CUDA_GRAPHS=1 runs the same goldens through captured
        // decode graphs; the ids must not change.
        cuda_graphs: std::env::var("VAPI_TEST_CUDA_GRAPHS").is_ok(),
    };
    CandleBackend::load(dir, &opts).expect("model loads")
}

fn scheduler(backend: &CandleBackend, prefix_cache: bool) -> Scheduler {
    let cfg = SchedulerConfig {
        max_concurrent_seqs: 8,
        max_batched_tokens: 512,
        // Small on purpose so a prompt is prefilled in several chunks.
        prefill_chunk_tokens: 24,
        max_context: backend.spec().max_context,
        block_size: BLOCK_SIZE,
        prefix_cache,
    };
    Scheduler::new(
        cfg,
        BlockPool::new(backend.num_blocks(), BLOCK_SIZE, 0),
        CacheNamespace::new("test", "fp", DType::F32, None, "global"),
        backend.spec().eos_token_ids.clone(),
    )
}

fn greedy(max_tokens: usize) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        max_tokens,
        ..Default::default()
    }
}

/// One generated token plus the logit margin it won by.
#[derive(Clone, Copy, Debug)]
struct Step {
    /// top-1 logit minus top-2 logit. Small means a near-tie, where bf16
    /// rounding can legitimately flip the choice.
    gap: f32,
}

struct Finished {
    generated: Vec<u32>,
    steps: Vec<Step>,
    reason: FinishReason,
}

/// Drive the scheduler until every admitted request finishes.
fn run(sched: &mut Scheduler, backend: &mut CandleBackend) -> HashMap<RequestId, Finished> {
    let mut steps: HashMap<SeqId, Vec<Step>> = HashMap::new();
    let mut out = HashMap::new();
    for _ in 0..10_000 {
        let plan = sched.schedule();
        if !plan.is_empty() {
            let logits = backend.forward(&plan.batch).expect("forward");
            let mut sampled = Vec::with_capacity(plan.sampled_seqs.len());
            for (row, seq_id) in plan.sampled_seqs.iter().enumerate() {
                let row = logits.row(row);
                let (mut best, mut second) = (0usize, f32::NEG_INFINITY);
                for (i, &x) in row.iter().enumerate() {
                    if x > row[best] {
                        second = row[best];
                        best = i;
                    } else if x > second {
                        second = x;
                    }
                }
                steps.entry(*seq_id).or_default().push(Step {
                    gap: row[best] - second,
                });
                sampled.push(best as u32);
            }
            sched.commit(&plan, &sampled);
        }
        for (id, seq) in sched.drain_finished() {
            let reason = match seq.status {
                SeqStatus::Finished(r) => r,
                _ => FinishReason::Error,
            };
            out.insert(
                seq.request_id.clone(),
                Finished {
                    generated: seq.generated().to_vec(),
                    steps: steps.remove(&id).unwrap_or_default(),
                    reason,
                },
            );
        }
        if sched.num_running() == 0 && sched.num_waiting() == 0 {
            break;
        }
    }
    out
}

/// Like `run`, but cancels `cancel.0` after `cancel.1` steps and counts how
/// many times a sequence was preempted.
fn run_with(
    sched: &mut Scheduler,
    backend: &mut CandleBackend,
    cancel: Option<(RequestId, usize)>,
) -> (HashMap<RequestId, Finished>, usize) {
    let mut steps: HashMap<SeqId, Vec<Step>> = HashMap::new();
    let mut out = HashMap::new();
    let mut preemptions = 0;
    for step in 0..10_000 {
        if let Some((rid, at)) = &cancel
            && *at == step
        {
            assert!(sched.cancel(rid).is_some(), "the request was still live");
        }
        let plan = sched.schedule();
        preemptions += plan.preempted.len();
        if !plan.is_empty() {
            let logits = backend.forward(&plan.batch).expect("forward");
            let mut sampled = Vec::with_capacity(plan.sampled_seqs.len());
            for (row, seq_id) in plan.sampled_seqs.iter().enumerate() {
                let row = logits.row(row);
                let (mut best, mut second) = (0usize, f32::NEG_INFINITY);
                for (i, &x) in row.iter().enumerate() {
                    if x > row[best] {
                        second = row[best];
                        best = i;
                    } else if x > second {
                        second = x;
                    }
                }
                steps.entry(*seq_id).or_default().push(Step {
                    gap: row[best] - second,
                });
                sampled.push(best as u32);
            }
            sched.commit(&plan, &sampled);
        }
        for (id, seq) in sched.drain_finished() {
            let reason = match seq.status {
                SeqStatus::Finished(r) => r,
                _ => FinishReason::Error,
            };
            out.insert(
                seq.request_id.clone(),
                Finished {
                    generated: seq.generated().to_vec(),
                    steps: steps.remove(&id).unwrap_or_default(),
                    reason,
                },
            );
        }
        if sched.num_running() == 0 && sched.num_waiting() == 0 {
            break;
        }
    }
    (out, preemptions)
}

/// Compare generated ids with the golden. Exact on the CPU; on a GPU the ids
/// may diverge at a near-tie, which is reported rather than failed.
fn check_against_golden(golden: &Golden, got: &Finished, exact: bool) {
    let n = golden.greedy_ids.len().min(got.generated.len());
    let first_diff = (0..n).find(|&i| golden.greedy_ids[i] != got.generated[i]);
    match first_diff {
        None if got.generated.len() == golden.greedy_ids.len() => {
            eprintln!("{}: {} tokens, exact match", golden.fixture, n);
        }
        None => panic!(
            "{}: lengths differ: vapi {} vs HF {} ({:?})",
            golden.fixture,
            got.generated.len(),
            golden.greedy_ids.len(),
            got.reason
        ),
        Some(i) => {
            let step = got.steps[i];
            eprintln!(
                "{}: diverges at token {i} of {}: vapi {} vs HF {}; logit gap there {:.4}",
                golden.fixture,
                golden.greedy_ids.len(),
                got.generated[i],
                golden.greedy_ids[i],
                step.gap
            );
            if exact {
                panic!(
                    "{}: CPU f32 must match HF exactly (expected {:?})",
                    golden.fixture, golden.greedy_text
                );
            }
            assert!(
                step.gap < 0.5,
                "{}: divergence at a logit gap of {:.3} is a bug, not rounding",
                golden.fixture,
                step.gap
            );
        }
    }
}

#[test]
fn the_chat_template_and_prompt_ids_match_hugging_face() {
    let Some(dir) = model_dir() else { return };
    let tok = TokenizerBundle::from_dir(&dir).expect("tokenizer loads");
    let template = tok.template().expect("model has a chat template");
    for g in goldens_for(&dir) {
        let messages: Vec<_> = g.messages.iter().map(|m| m.to_chat()).collect();
        let rendered = template
            .render_with(
                &messages,
                true,
                tok.bos_token.as_deref().unwrap_or(""),
                tok.eos_token.as_deref().unwrap_or(""),
                g.tools.as_deref(),
            )
            .unwrap();
        assert_eq!(rendered, g.rendered, "{}: template text", g.fixture);
        let ids = tok.encode_chat(&messages, g.tools.as_deref()).unwrap();
        assert_eq!(ids, g.prompt_ids, "{}: prompt ids", g.fixture);
    }
}

#[test]
fn greedy_decoding_matches_hugging_face() {
    let Some(dir) = model_dir() else { return };
    let mut backend = load(&dir);
    let exact = !backend.device().is_cuda();
    for g in goldens_for(&dir) {
        let mut sched = scheduler(&backend, true);
        let rid = RequestId::new();
        sched
            .admit(
                rid.clone(),
                g.prompt_ids.clone(),
                greedy(g.max_new_tokens),
                "global".into(),
            )
            .unwrap();
        let done = run(&mut sched, &mut backend);
        check_against_golden(&g, &done[&rid], exact);
        assert_eq!(
            sched.pool().stats().num_free,
            backend.num_blocks(),
            "{}: blocks leaked",
            g.fixture
        );
    }
}

#[test]
fn concurrent_requests_each_get_their_own_answer() {
    // Step 4's check: several requests in flight at once through the real
    // backend, mixed prefill and decode in one batch, every block returned.
    let Some(dir) = model_dir() else { return };
    let mut backend = load(&dir);
    let exact = !backend.device().is_cuda();
    let goldens = goldens_for(&dir);
    let mut sched = scheduler(&backend, true);
    let mut rids = Vec::new();
    for g in &goldens {
        let rid = RequestId::new();
        sched
            .admit(
                rid.clone(),
                g.prompt_ids.clone(),
                greedy(g.max_new_tokens),
                "global".into(),
            )
            .unwrap();
        rids.push(rid);
    }
    let done = run(&mut sched, &mut backend);
    assert_eq!(done.len(), goldens.len());
    for (g, rid) in goldens.iter().zip(&rids) {
        check_against_golden(g, &done[rid], exact);
    }
    sched.pool().check_invariants();
    assert_eq!(
        sched.pool().stats().num_free,
        backend.num_blocks(),
        "blocks leaked"
    );
}

#[test]
fn the_prefix_cache_does_not_change_the_output() {
    // The real-model version of `prefix_cache_hits_do_not_change_the_output`:
    // the second request hits the first one's blocks and must still produce
    // the same tokens as a run with the cache off.
    let Some(dir) = model_dir() else { return };
    let mut backend = load(&dir);
    let goldens = goldens_for(&dir);
    let g = goldens
        .iter()
        .find(|g| g.fixture == "multiturn")
        .unwrap_or(&goldens[0]);
    let model_id = &g.model_id;

    let mut cold = scheduler(&backend, false);
    let rid = RequestId::new();
    cold.admit(
        rid.clone(),
        g.prompt_ids.clone(),
        greedy(32),
        "global".into(),
    )
    .unwrap();
    let without_cache = run(&mut cold, &mut backend).remove(&rid).unwrap().generated;

    let mut warm = scheduler(&backend, true);
    let first = RequestId::new();
    warm.admit(
        first.clone(),
        g.prompt_ids.clone(),
        greedy(32),
        "global".into(),
    )
    .unwrap();
    let a = run(&mut warm, &mut backend)
        .remove(&first)
        .unwrap()
        .generated;
    let second = RequestId::new();
    warm.admit(
        second.clone(),
        g.prompt_ids.clone(),
        greedy(32),
        "global".into(),
    )
    .unwrap();
    let b = run(&mut warm, &mut backend)
        .remove(&second)
        .unwrap()
        .generated;

    let stats = warm.pool().stats();
    assert!(
        stats.hit_tokens > 0,
        "the second request must have hit the prefix cache ({model_id})"
    );
    assert_eq!(a, without_cache, "first (cold) run vs cache off");
    assert_eq!(b, without_cache, "cache-hit run vs cache off");
    assert_eq!(stats.num_free, backend.num_blocks(), "blocks leaked");
}

/// Two requests whose blocks together exceed the cache: one is preempted,
/// its blocks are taken, and it is recomputed once the other finishes. The
/// recomputed sequence must produce the same tokens as an uninterrupted
/// run, which is the golden.
#[test]
fn a_preempted_request_recomputes_to_the_same_answer() {
    let Some(dir) = model_dir() else { return };
    let goldens = goldens_for(&dir);
    let pick = |name: &str| goldens.iter().find(|g| g.fixture == name).expect(name);
    // The two longest generations: both prompts fit at the start, and the
    // cache is two blocks short of holding both at their full length, so a
    // decode step runs out of blocks while both are still running and one
    // sequence is preempted. (One short is not enough: the shorter answer
    // can finish and free its blocks the step before the other needs one.)
    let (a, b) = (pick("multiturn"), pick("system"));
    let need = |g: &Golden| (g.prompt_ids.len() + g.greedy_ids.len()).div_ceil(BLOCK_SIZE);
    let prompt_blocks = |g: &Golden| g.prompt_ids.len().div_ceil(BLOCK_SIZE);
    let blocks = need(a) + need(b) - 2;
    assert!(prompt_blocks(a) + prompt_blocks(b) < blocks);
    // The backend reserves a block or two of its own (padding rows for
    // CUDA graphs); the scheduler's pool is what decides preemption, so
    // size that exactly and only require the backend to hold at least it.
    let mut backend = load_with(&dir, blocks + 2);
    assert!(backend.num_blocks() >= blocks);
    let mut sched = {
        let cfg = SchedulerConfig {
            max_concurrent_seqs: 8,
            max_batched_tokens: 512,
            prefill_chunk_tokens: 24,
            max_context: backend.spec().max_context,
            block_size: BLOCK_SIZE,
            prefix_cache: false,
        };
        Scheduler::new(
            cfg,
            BlockPool::new(blocks, BLOCK_SIZE, 0),
            CacheNamespace::new("test", "fp", DType::F32, None, "global"),
            backend.spec().eos_token_ids.clone(),
        )
    };
    let mut ids: Vec<RequestId> = Vec::new();
    for g in [a, b] {
        let rid = RequestId::new();
        sched
            .admit(
                rid.clone(),
                g.prompt_ids.clone(),
                greedy(g.max_new_tokens),
                "global".into(),
            )
            .expect("admit");
        ids.push(rid);
    }
    let (out, preemptions) = run_with(&mut sched, &mut backend, None);
    assert!(
        preemptions >= 1,
        "the cache was sized so that a preemption must happen"
    );
    let exact = backend.device().is_cpu();
    for (g, rid) in [a, b].iter().zip(&ids) {
        let got = out.get(rid).expect("finished");
        assert_ne!(got.reason, FinishReason::Error);
        check_against_golden(g, got, exact);
    }
    assert_eq!(
        sched.pool().num_free(),
        blocks,
        "every block is back once both are done"
    );
}

/// Cancelling one request mid-generation frees its blocks and leaves the
/// other request's output untouched.
#[test]
fn cancelling_one_request_does_not_disturb_the_other() {
    let Some(dir) = model_dir() else { return };
    let goldens = goldens_for(&dir);
    let pick = |name: &str| goldens.iter().find(|g| g.fixture == name).expect(name);
    let (keep, drop) = (pick("multiturn"), pick("system"));
    let mut backend = load(&dir);
    let mut sched = scheduler(&backend, false);
    let mut admit = |g: &Golden| {
        let rid = RequestId::new();
        sched
            .admit(
                rid.clone(),
                g.prompt_ids.clone(),
                greedy(g.max_new_tokens),
                "global".into(),
            )
            .expect("admit");
        rid
    };
    let keep_rid = admit(keep);
    let drop_rid = admit(drop);
    let (out, _) = run_with(&mut sched, &mut backend, Some((drop_rid.clone(), 12)));
    let dropped = out
        .get(&drop_rid)
        .expect("the cancelled request is reported");
    assert_eq!(dropped.reason, FinishReason::Cancelled);
    assert!(
        dropped.generated.len() < drop.max_new_tokens,
        "it was cut short: {} tokens",
        dropped.generated.len()
    );
    let kept = out.get(&keep_rid).expect("the other request finished");
    check_against_golden(keep, kept, backend.device().is_cpu());
    assert_eq!(
        sched.pool().num_free(),
        backend.num_blocks(),
        "no block leaked"
    );
}
