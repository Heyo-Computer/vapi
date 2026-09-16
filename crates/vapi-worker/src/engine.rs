use std::collections::HashMap;

use vapi_cache::{BlockPool, CacheNamespace};
use vapi_core::{Config, FinishReason, RequestId, Result, SeqId};
use vapi_engine::backend::ExecutionBackend;
use vapi_engine::{
    IncrementalDetokenizer, Sampler, SamplerState, Scheduler, SchedulerConfig, SeqStatus,
};
use vapi_proto::{Delta, DeltaSeq, Job};
use vapi_tokenize::Tokenization;

/// Everything produced by one engine step, for the caller to publish.
pub struct StepOutput {
    pub events: Vec<(RequestId, Delta)>,
    /// Requests that reached a terminal state this step; their JetStream
    /// messages can now be acked.
    pub completed: Vec<RequestId>,
    pub did_work: bool,
}

/// Owns the scheduler, the backend and all per-sequence streaming state.
///
/// Deliberately not `async`: it is a synchronous state machine driven one step
/// at a time by the worker's event loop, which keeps the CPU-bound forward
/// pass out of the async runtime's way and makes the whole thing testable
/// without a runtime at all.
pub struct Engine {
    scheduler: Scheduler,
    backend: Box<dyn ExecutionBackend>,
    tokenizer: Tokenization,
    detok: HashMap<SeqId, IncrementalDetokenizer>,
    samplers: HashMap<SeqId, SamplerState>,
    seq_to_request: HashMap<SeqId, RequestId>,
    delta_seq: HashMap<RequestId, DeltaSeq>,
    worker_id: String,
}

impl Engine {
    pub fn new(
        cfg: &Config,
        backend: Box<dyn ExecutionBackend>,
        tokenizer: Tokenization,
        worker_id: String,
    ) -> Self {
        let spec = backend.spec().clone();
        let namespace = CacheNamespace::new(
            &cfg.model.id,
            // Until real weights are loaded there is nothing to fingerprint;
            // the model id alone still keeps separate models apart.
            "unversioned",
            cfg.model.dtype,
            None,
            &cfg.cache.default_namespace,
        );
        let num_blocks = backend.num_blocks();
        let watermark = ((num_blocks as f32) * cfg.cache.watermark).ceil() as usize;

        let sched_cfg = SchedulerConfig {
            max_concurrent_seqs: cfg.worker.max_concurrent_seqs,
            max_batched_tokens: cfg.worker.max_batched_tokens,
            prefill_chunk_tokens: cfg.worker.prefill_chunk_tokens,
            max_context: cfg.model.max_context.min(spec.max_context),
            block_size: vapi_cache::BLOCK_SIZE,
            prefix_cache: cfg.cache.prefix_cache,
        };
        let pool = BlockPool::new(num_blocks, vapi_cache::BLOCK_SIZE, watermark);
        let eos = if spec.eos_token_ids.is_empty() {
            tokenizer.eos_token_ids()
        } else {
            spec.eos_token_ids.clone()
        };

        Self {
            scheduler: Scheduler::new(sched_cfg, pool, namespace, eos),
            backend,
            tokenizer,
            detok: HashMap::new(),
            samplers: HashMap::new(),
            seq_to_request: HashMap::new(),
            delta_seq: HashMap::new(),
            worker_id,
        }
    }

    pub fn headroom(&self) -> usize {
        self.scheduler.admission_headroom()
    }

    pub fn is_idle(&self) -> bool {
        self.scheduler.num_running() == 0
            && self.scheduler.num_waiting() == 0
            // A sequence cancelled between steps is finished but not yet
            // reported; going idle now would strand its Done event.
            && !self.scheduler.has_finished()
    }

    pub fn stats(&self) -> (usize, usize, f32, f32) {
        let s = self.scheduler.pool().stats();
        (
            self.scheduler.num_running(),
            self.scheduler.num_waiting(),
            s.utilization(),
            s.hit_rate(),
        )
    }

    pub fn admit(&mut self, job: Job) -> Result<SeqId> {
        let rid = job.request_id.clone();
        let seed = job.params.seed;
        let prompt = job.prompt_tokens;
        let id = self
            .scheduler
            .admit(rid.clone(), prompt.clone(), job.params, job.namespace)?;
        self.detok
            .insert(id, IncrementalDetokenizer::with_prompt(&prompt));
        self.samplers.insert(id, SamplerState::new(seed));
        self.seq_to_request.insert(id, rid.clone());
        self.delta_seq.insert(rid, DeltaSeq::default());
        Ok(id)
    }

    pub fn cancel(&mut self, request_id: &RequestId) -> bool {
        self.scheduler.cancel(request_id).is_some()
    }

    /// Run one scheduler step.
    pub fn step(&mut self) -> Result<StepOutput> {
        let mut out = StepOutput {
            events: Vec::new(),
            completed: Vec::new(),
            did_work: false,
        };

        let plan = self.scheduler.schedule();

        for (seq_id, cached) in &plan.started {
            if let Some(rid) = self.seq_to_request.get(seq_id) {
                let delta = Delta::Started {
                    worker_id: self.worker_id.clone(),
                    cached_prefix_tokens: *cached,
                };
                out.events.push((rid.clone(), delta));
            }
        }

        if !plan.is_empty() {
            out.did_work = true;
            let logits = self.backend.forward(&plan.batch)?;

            let mut sampled = Vec::with_capacity(plan.sampled_seqs.len());
            for (row, seq_id) in plan.sampled_seqs.iter().enumerate() {
                let params = self
                    .scheduler
                    .get(*seq_id)
                    .map(|s| s.params.clone())
                    .unwrap_or_default();
                let state = self
                    .samplers
                    .entry(*seq_id)
                    .or_insert_with(|| SamplerState::new(params.seed));
                let mut row_logits = logits.row(row).to_vec();
                let token = Sampler::sample(&mut row_logits, &params, state);
                state.observe(token);
                sampled.push(token);
            }

            // Emit text before committing, so a token that turns out to be EOS
            // is not streamed to the client as content.
            let eos = self.eos_ids();
            for (seq_id, &token) in plan.sampled_seqs.iter().zip(&sampled) {
                if eos.contains(&token) {
                    continue;
                }
                let Some(rid) = self.seq_to_request.get(seq_id).cloned() else {
                    continue;
                };
                let text = {
                    let tokenizer = &self.tokenizer;
                    let d = self.detok.entry(*seq_id).or_default();
                    d.push(token, |ids| tokenizer.decode(ids))
                };
                if !text.is_empty() {
                    out.events.push((
                        rid,
                        Delta::Token {
                            text,
                            token_id: token,
                            logprob: None,
                        },
                    ));
                }
            }

            self.scheduler.commit(&plan, &sampled);
        }

        for seq_id in &plan.preempted {
            tracing::debug!(?seq_id, "preempted for lack of blocks");
        }

        for (seq_id, seq) in self.scheduler.drain_finished() {
            let reason = match seq.status {
                SeqStatus::Finished(r) => r,
                _ => FinishReason::Error,
            };
            if let Some(rid) = self.seq_to_request.remove(&seq_id) {
                out.events.push((
                    rid.clone(),
                    Delta::Done {
                        reason,
                        prompt_tokens: seq.prompt_len(),
                        completion_tokens: seq.num_generated(),
                    },
                ));
                out.completed.push(rid);
            }
            self.detok.remove(&seq_id);
            self.samplers.remove(&seq_id);
            out.did_work = true;
        }

        Ok(out)
    }

    fn eos_ids(&self) -> Vec<u32> {
        let spec = self.backend.spec();
        if spec.eos_token_ids.is_empty() {
            self.tokenizer.eos_token_ids()
        } else {
            spec.eos_token_ids.clone()
        }
    }

    /// Stamp a delta with its position in the request's stream.
    pub fn sequence(&mut self, rid: &RequestId, delta: Delta) -> vapi_proto::DeltaMsg {
        self.delta_seq.entry(rid.clone()).or_default().next(delta)
    }

    pub fn forget(&mut self, rid: &RequestId) {
        self.delta_seq.remove(rid);
    }
}
