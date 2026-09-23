use std::collections::HashMap;

use vapi_cache::{BlockPool, CacheNamespace, SpillStore, hash_block_chain};
use vapi_core::{Config, FinishReason, RequestId, ResponseFormat, Result, SeqId, StopCondition};
use vapi_engine::backend::ExecutionBackend;
use vapi_engine::{
    CandidateNeed, IncrementalDetokenizer, Machine, RowLogits, RowLogprobs, Sampler, SamplerState,
    Scheduler, SchedulerConfig, Schema, SeqStatus, StepLogits, TokenMasker,
};
use vapi_proto::{Delta, DeltaSeq, Job, TokenLogprob};
use vapi_tokenize::Tokenization;

/// Everything produced by one engine step, for the caller to publish.
#[derive(Debug)]
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
    streams: HashMap<SeqId, StreamState>,
    seq_to_request: HashMap<SeqId, RequestId>,
    /// Which completion a sequence is producing, for `n > 1`.
    choice_of: HashMap<SeqId, u32>,
    /// Requests whose leader has yet to fork, and into how many.
    pending_forks: HashMap<SeqId, usize>,
    /// Choices still running per request, so the JetStream message is
    /// acked once, when the last one is done.
    open_choices: HashMap<RequestId, usize>,
    delta_seq: HashMap<RequestId, DeltaSeq>,
    worker_id: String,
    steps: u64,
    /// Tier 3. `None` when it is off or the backend cannot move blocks.
    spill: Option<SpillStore>,
    /// Vocabulary index for constrained decoding, built on first use: it
    /// costs a pass over every token's text, which a server that never
    /// sees a `response_format` should not pay.
    masker: Option<std::sync::Arc<TokenMasker>>,
    /// Compiled schemas, keyed by the schema text, so a repeated request
    /// shape is compiled once.
    schemas: HashMap<String, std::sync::Arc<Schema>>,
    /// The namespace every block hash is taken under, needed to look a
    /// prompt up in the spill tier before the scheduler sees it.
    namespace: CacheNamespace,
}

/// Per-sequence streaming state: detokenizer, RNG, and the text held back
/// while a stop string might still complete.
struct StreamState {
    detok: IncrementalDetokenizer,
    sampler: SamplerState,
    /// Decoded text not yet sent, because its tail could be the start of a
    /// stop string that the next token completes. Empty when the request has
    /// no stop strings.
    held: String,
    /// Last token sampled, so a flush of `held` at completion can still carry
    /// a token id.
    last_token: u32,
    /// A stop string matched this step; the scheduler is told after the
    /// emission pass.
    stopped: bool,
    /// Constrained decoding state, when the request asked for a format.
    constraint: Option<Constraint>,
}

/// A response-format constraint, before or after it takes effect.
#[derive(Clone)]
enum Constraint {
    /// Waiting for `marker` in the output. The model is unconstrained
    /// until then, which is what lets a reasoning model think first.
    Waiting {
        marker: String,
        /// The marker as a single token, when the vocabulary has one. It
        /// is what the model is made to emit when it tries to end its turn
        /// before producing the document.
        close_token: Option<u32>,
        /// The tail of the output so far, long enough to catch a marker
        /// split across tokens.
        tail: String,
        schema: std::sync::Arc<Schema>,
    },
    Active(Machine),
}

/// What one row's constraint does to its logits this step.
enum RowConstraint {
    /// The document is being written: mask everything that would break it.
    /// `finishing` means the budget is nearly gone, so only tokens that
    /// close what is open are left.
    Active { machine: Machine, finishing: bool },
    /// Still thinking. The model writes freely, but it may not end its
    /// turn: when it tries to, it is made to close the reasoning block
    /// instead, which is what starts the document.
    ///
    /// Masking the end of turn alone is not enough. LFM2.5 drafts its
    /// answer inside the reasoning block and then wants to stop; with the
    /// stop masked it carries on thinking and never closes the block, so
    /// the caller gets a full budget of reasoning and no answer.
    Waiting {
        close_token: Option<u32>,
        /// The reasoning has run long enough: close it now. Without a cap a
        /// model that thinks past its budget returns a truncated monologue
        /// and no document, which is the one outcome a caller asking for a
        /// format cannot use.
        out_of_patience: bool,
    },
}

impl Constraint {
    fn machine(&self) -> Option<&Machine> {
        match self {
            Self::Active(m) => Some(m),
            Self::Waiting { .. } => None,
        }
    }

    fn row(&self, out_of_patience: bool) -> RowConstraint {
        match self {
            Self::Active(m) => RowConstraint::Active {
                machine: m.clone(),
                finishing: false,
            },
            Self::Waiting { close_token, .. } => RowConstraint::Waiting {
                close_token: *close_token,
                out_of_patience,
            },
        }
    }

    /// Advance with the text of one sampled token. `false` means an active
    /// machine rejected it, which would be a bug in the mask.
    fn push(&mut self, text: &str) -> bool {
        match self {
            Self::Active(m) => m.push(text),
            Self::Waiting {
                marker,
                tail,
                schema,
                ..
            } => {
                tail.push_str(text);
                let Some(at) = tail.find(marker.as_str()) else {
                    // Keep only what could still be the start of a marker.
                    let keep = marker.len().saturating_sub(1);
                    if tail.len() > keep {
                        let cut = tail.len() - keep;
                        let cut = (0..=cut).rev().find(|i| tail.is_char_boundary(*i));
                        if let Some(cut) = cut {
                            tail.drain(..cut);
                        }
                    }
                    return true;
                };
                // Everything after the marker is already part of the
                // answer, so the machine has to see it.
                let rest = tail[at + marker.len()..].to_string();
                let mut m = Machine::new(schema.clone());
                let ok = m.push(rest.trim_start());
                *self = Self::Active(m);
                ok
            }
        }
    }
}

/// Tokens reserved for closing a constrained document. Deep nesting needs
/// one per level plus the odd string terminator; sixteen is generous for
/// the schemas this supports.
const CLOSING_BUDGET: usize = 16;

/// The highest-scoring token of a row, or `None` for an empty row.
fn argmax(logits: &[f32]) -> Option<u32> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &l) in logits.iter().enumerate() {
        if best.is_none_or(|(_, b)| l > b) {
            best = Some((i, l));
        }
    }
    best.map(|(i, _)| i as u32)
}

/// Decide what to stream now, given newly decoded `text` and the request's
/// stop strings.
///
/// With no stop strings everything goes out immediately. Otherwise text is
/// accumulated in `held`, and only the part that can no longer be the start of
/// a stop string is released: the last `max_stop_len - 1` bytes stay back,
/// because the next token could complete a match across the boundary. When a
/// match is found the text is truncated there and the sequence marked stopped.
fn emit_with_stop_strings(
    st: &mut StreamState,
    stop: &StopCondition,
    text: String,
) -> Option<String> {
    if stop.stop_strings.is_empty() {
        return (!text.is_empty()).then_some(text);
    }
    st.held.push_str(&text);
    if let Some(cut) = stop.find_stop_string(&st.held) {
        let out = st.held[..cut].to_string();
        st.held.clear();
        st.stopped = true;
        return (!out.is_empty()).then_some(out);
    }
    let keep = stop.max_stop_len().saturating_sub(1);
    if st.held.len() <= keep {
        return None;
    }
    let mut cut = st.held.len() - keep;
    while !st.held.is_char_boundary(cut) {
        cut -= 1;
    }
    if cut == 0 {
        return None;
    }
    let out: String = st.held.drain(..cut).collect();
    Some(out)
}

impl Engine {
    pub fn new(
        cfg: &Config,
        backend: Box<dyn ExecutionBackend>,
        tokenizer: Tokenization,
        worker_id: String,
    ) -> Self {
        let spec = backend.spec().clone();
        // The fingerprint keeps one version's KV from being served as
        // another's when weights change under the same model id, and it is
        // what makes persisted blocks (tier 3) safe across restarts.
        let fingerprint = cfg
            .model
            .path
            .as_deref()
            .map(vapi_backend_candle::weights_fingerprint)
            .unwrap_or_else(|| "unversioned".into());
        let namespace = CacheNamespace::new(
            &cfg.model.id,
            &fingerprint,
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
        let mut pool = BlockPool::new(num_blocks, vapi_cache::BLOCK_SIZE, watermark);

        // Tier 3 needs a backend that can move blocks; without one it stays
        // off rather than failing at the first eviction.
        let block_bytes = backend.block_bytes();
        let spill = match (cfg.cache.spill, block_bytes) {
            (true, 0) => {
                tracing::warn!("cache.spill is on but this backend cannot move blocks; ignoring");
                None
            }
            (true, _) => {
                let dir = (cfg.cache.spill_max_bytes > 0).then_some(cfg.cache.spill_dir.as_path());
                match SpillStore::open(dir, cfg.cache.spill_ram_bytes, cfg.cache.spill_max_bytes) {
                    Ok(store) => {
                        tracing::info!(
                            block_bytes,
                            ram_max = cfg.cache.spill_ram_bytes,
                            disk_max = cfg.cache.spill_max_bytes,
                            held = store.stats().disk_blocks,
                            "spill tier ready"
                        );
                        pool.track_evictions(true);
                        Some(store)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "spill tier unavailable; running without it");
                        None
                    }
                }
            }
            (false, _) => None,
        };
        let eos = if spec.eos_token_ids.is_empty() {
            tokenizer.eos_token_ids()
        } else {
            spec.eos_token_ids.clone()
        };

        Self {
            scheduler: Scheduler::new(sched_cfg, pool, namespace.clone(), eos),
            backend,
            tokenizer,
            streams: HashMap::new(),
            seq_to_request: HashMap::new(),
            choice_of: HashMap::new(),
            pending_forks: HashMap::new(),
            open_choices: HashMap::new(),
            delta_seq: HashMap::new(),
            worker_id,
            steps: 0,
            spill,
            namespace,
            masker: None,
            schemas: HashMap::new(),
        }
    }

    /// The vocabulary index, built on first use.
    fn masker(&mut self) -> std::sync::Arc<TokenMasker> {
        if let Some(m) = &self.masker {
            return m.clone();
        }
        let t = std::time::Instant::now();
        let vocab = self.backend.spec().vocab_size;
        let texts: Vec<String> = (0..vocab as u32)
            .map(|id| self.tokenizer.decode(&[id]))
            .collect();
        let masker = std::sync::Arc::new(TokenMasker::new(texts));
        tracing::info!(
            vocab,
            ms = format_args!("{:.0}", t.elapsed().as_secs_f64() * 1e3),
            "built the constrained-decoding vocabulary index"
        );
        self.masker = Some(masker.clone());
        masker
    }

    /// Compile (or reuse) the schema a request asks for.
    fn compile_format(&mut self, format: &ResponseFormat) -> Result<std::sync::Arc<Schema>> {
        let key = match format {
            ResponseFormat::JsonObject => "{}".to_string(),
            ResponseFormat::JsonSchema { schema } => schema.to_string(),
        };
        if let Some(s) = self.schemas.get(&key) {
            return Ok(s.clone());
        }
        let compiled = match format {
            ResponseFormat::JsonObject => Schema::any(),
            ResponseFormat::JsonSchema { schema } => vapi_engine::structured::with_any_node(
                Schema::compile(schema).map_err(vapi_core::Error::InvalidRequest)?,
            ),
        };
        let compiled = std::sync::Arc::new(compiled);
        self.schemas.insert(key, compiled.clone());
        Ok(compiled)
    }

    /// Preserve the content of blocks the pool just evicted.
    ///
    /// Must run before anything writes to them: the pool has already handed
    /// them out, so the window is exactly "after the allocation, before the
    /// forward".
    fn spill_evicted(&mut self) {
        let Some(store) = &mut self.spill else { return };
        let evicted = self.scheduler.pool_mut().take_evicted();
        if evicted.is_empty() {
            return;
        }
        let t = std::time::Instant::now();
        let mut n = 0;
        for (hash, id) in evicted {
            if store.contains(&hash) {
                continue;
            }
            match self.backend.export_block(id) {
                Ok(bytes) => {
                    store.put(hash, bytes);
                    n += 1;
                }
                Err(e) => tracing::warn!(error = %e, "could not export an evicted block"),
            }
        }
        if n > 0 {
            metrics::histogram!("vapi_spill_export_ms")
                .record(t.elapsed().as_secs_f64() * 1e3 / n as f64);
        }
    }

    /// Put back whatever of this prompt the spill tier still holds, so the
    /// scheduler's prefix match finds it resident.
    ///
    /// Only the contiguous prefix is worth restoring, for the same reason the
    /// prefix cache only matches one: block `k`'s KV depends on `0..k`.
    fn promote_from_spill(&mut self, tokens: &[u32]) {
        if self.spill.is_none() {
            return;
        }
        let block_size = vapi_cache::BLOCK_SIZE;
        let full_blocks = tokens.len() / block_size;
        if full_blocks == 0 {
            return;
        }
        let hashes = hash_block_chain(
            &self.namespace,
            &tokens[..full_blocks * block_size],
            block_size,
        );
        // A restored block is one the sequence itself will use, so this does
        // not compete with its admission; keep one block in hand so the pool
        // can never be emptied by promotion alone.
        let margin = 1;
        let t = std::time::Instant::now();
        let mut restored = 0usize;
        for hash in &hashes {
            if self.scheduler.pool().contains(hash) {
                continue;
            }
            let Some(store) = &mut self.spill else { break };
            let Some(bytes) = store.get(hash) else { break };
            if self.scheduler.pool().num_free() <= margin {
                break;
            }
            let Ok(id) = self.scheduler.pool_mut().allocate() else {
                break;
            };
            // The allocation may have evicted a cached block; that content
            // has to leave before the import overwrites it.
            self.spill_evicted();
            if let Err(e) = self.backend.import_block(id, &bytes) {
                tracing::warn!(error = %e, "could not import a spilled block");
                self.scheduler.pool_mut().free(id);
                break;
            }
            let pool = self.scheduler.pool_mut();
            let canonical = pool.publish(id, *hash);
            if canonical != id {
                pool.free(id);
            }
            // Resident but unreferenced: exactly what a prefix-cache hit
            // expects to find.
            pool.free(canonical);
            restored += 1;
        }
        if restored > 0 {
            metrics::histogram!("vapi_spill_import_ms")
                .record(t.elapsed().as_secs_f64() * 1e3 / restored as f64);
            metrics::counter!("vapi_spill_blocks_restored_total").increment(restored as u64);
            tracing::debug!(restored, "restored blocks from the spill tier");
        }
    }

    /// Turn one prefilled sequence into `extra` more, each sampling its
    /// own first token from the row the leader just produced.
    ///
    /// The prompt is computed once and shared: full blocks by reference,
    /// the partial last block copied, since every choice writes a
    /// different token into its next slot.
    fn fork_request(
        &mut self,
        leader: SeqId,
        extra: usize,
        row: &[f32],
        params: &vapi_core::SamplingParams,
        out: &mut StepOutput,
    ) -> Result<()> {
        let Some(rid) = self.seq_to_request.get(&leader).cloned() else {
            return Ok(());
        };
        let fork = self.scheduler.fork(leader, extra)?;
        if !fork.copies.is_empty() {
            self.backend.copy_blocks(&fork.copies)?;
        }
        metrics::counter!("vapi_forks_total").increment(fork.ids.len() as u64);
        let prompt: Vec<u32> = self
            .scheduler
            .get(leader)
            .map(|s| s.tokens()[..s.prompt_len()].to_vec())
            .unwrap_or_default();
        for (i, id) in fork.ids.iter().enumerate() {
            let choice = i as u32 + 1;
            // A different seed per choice, or the sampling would repeat the
            // leader's draw for a seeded request.
            let mut state = SamplerState::new(params.seed_for(choice as usize));
            let mut row = row.to_vec();
            let token = Sampler::sample(&mut row, params, &mut state);
            state.observe(token);
            self.scheduler.seed_fork(*id, token)?;
            let mut st = StreamState {
                detok: IncrementalDetokenizer::with_prompt(&prompt),
                sampler: state,
                held: String::new(),
                last_token: token,
                stopped: false,
                constraint: None,
            };
            // The choice's first token is emitted here, as the leader's was
            // by the sampling loop.
            let tokenizer = &self.tokenizer;
            let text = st.detok.push(token, |ids| tokenizer.decode(ids));
            let eos = self.eos_ids();
            if !eos.contains(&token)
                && let Some(text) = emit_with_stop_strings(&mut st, &params.stop, text)
            {
                out.events.push((
                    rid.clone(),
                    Delta::Token {
                        choice,
                        text,
                        token_id: token,
                        logprob: None,
                        top_logprobs: None,
                    },
                ));
            }
            self.streams.insert(*id, st);
            self.seq_to_request.insert(*id, rid.clone());
            self.choice_of.insert(*id, choice);
        }
        Ok(())
    }

    /// Spill-tier counters, for tests and the worker's gauges.
    pub fn spill_stats(&self) -> Option<vapi_cache::SpillStats> {
        self.spill.as_ref().map(|s| s.stats())
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
        let params = job.params.clone();
        let prompt = job.prompt_tokens;
        // Compile the constraint before anything is allocated: an
        // unsupported schema should fail the request, not a step.
        let constraint = match &job.params.response_format {
            None => None,
            Some(f) => {
                let schema = self.compile_format(f)?;
                // Building the index can take a moment on a large
                // vocabulary; do it now rather than inside the first step.
                let _ = self.masker();
                Some(match &job.params.constraint_starts_after {
                    Some(marker) if !marker.is_empty() => Constraint::Waiting {
                        close_token: self.tokenizer.token_id(marker),
                        marker: marker.clone(),
                        tail: String::new(),
                        schema,
                    },
                    _ => Constraint::Active(Machine::new(schema)),
                })
            }
        };
        // Before the scheduler matches the prefix cache, give it back
        // anything this prompt needs that only the spill tier still has.
        self.promote_from_spill(&prompt);
        let id = self
            .scheduler
            .admit(rid.clone(), prompt.clone(), job.params, job.namespace)?;
        self.streams.insert(
            id,
            StreamState {
                detok: IncrementalDetokenizer::with_prompt(&prompt),
                sampler: SamplerState::new(seed),
                held: String::new(),
                last_token: 0,
                stopped: false,
                constraint,
            },
        );
        self.seq_to_request.insert(id, rid.clone());
        self.choice_of.insert(id, 0);
        // Greedy choices would be identical, so `n` only forks when the
        // request actually samples.
        let choices = if params.is_greedy() {
            1
        } else {
            params.n.max(1)
        };
        if choices > 1 {
            self.pending_forks.insert(id, choices - 1);
        }
        self.open_choices.insert(rid.clone(), choices);
        self.delta_seq.insert(rid, DeltaSeq::default());
        Ok(id)
    }

    pub fn cancel(&mut self, request_id: &RequestId) -> bool {
        self.scheduler.cancel(request_id).is_some()
    }

    /// Fail every sequence the engine holds, freeing their blocks, and
    /// return the `Failed` event for each. The failure policy after a step
    /// error, and the end of a drain that ran out of time: a client gets an
    /// error frame it can retry on, never a stream that stops mid-sentence.
    pub fn fail_all(&mut self, message: &str) -> StepOutput {
        let mut out = StepOutput {
            events: Vec::new(),
            completed: Vec::new(),
            did_work: true,
        };
        let rids: Vec<RequestId> = self.seq_to_request.values().cloned().collect();
        for rid in &rids {
            self.scheduler.cancel(rid);
        }
        for (seq_id, _) in self.scheduler.drain_finished() {
            self.streams.remove(&seq_id);
            self.seq_to_request.remove(&seq_id);
        }
        // Anything the scheduler never knew about (should be nothing) is
        // still answered.
        self.streams.clear();
        self.seq_to_request.clear();
        for rid in rids {
            out.events.push((
                rid.clone(),
                Delta::Failed {
                    message: message.to_string(),
                },
            ));
            out.completed.push(rid);
        }
        out
    }

    /// Run one scheduler step.
    pub fn step(&mut self) -> Result<StepOutput> {
        let mut out = StepOutput {
            events: Vec::new(),
            completed: Vec::new(),
            did_work: false,
        };

        let t0 = std::time::Instant::now();
        let plan = self.scheduler.schedule();
        let t_schedule = t0.elapsed();
        // Scheduling allocates, which can evict cached blocks; their content
        // has to be preserved before the forward overwrites it.
        self.spill_evicted();

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
            let t1 = std::time::Instant::now();
            // Greedy, penalty-free rows need only the argmax, which the
            // backend can produce on the device; anything else needs the
            // full distribution on the host.
            let all_greedy = plan.sampled_seqs.iter().all(|id| {
                self.scheduler
                    .get(*id)
                    .map(|s| {
                        s.params.is_greedy()
                            && s.params.repetition_penalty == 1.0
                            && s.params.frequency_penalty == 0.0
                            && s.params.presence_penalty == 0.0
                            && s.params.logprobs.is_none()
                            && s.params.response_format.is_none()
                    })
                    .unwrap_or(false)
            });
            let device_sampled = if all_greedy {
                self.backend.forward_greedy(&plan.batch)?
            } else {
                None
            };
            let mut logits = match device_sampled {
                Some(_) => None,
                None => {
                    // Ask the backend for each row's candidates only; it
                    // returns the full logits when it cannot do that exactly.
                    let needs: Vec<CandidateNeed> = plan
                        .sampled_seqs
                        .iter()
                        .map(|id| {
                            let params = self
                                .scheduler
                                .get(*id)
                                .map(|s| s.params.clone())
                                .unwrap_or_default();
                            let mut need = self
                                .streams
                                .get(id)
                                .map(|st| st.sampler.need(&params))
                                .unwrap_or(CandidateNeed {
                                    inv_temperature: 1.0,
                                    needed: 0,
                                    gumbel: None,
                                    full: false,
                                });
                            // A row that is about to fork needs its whole
                            // distribution on the host: every choice draws
                            // its first token from it, and the device
                            // shortcuts return one token or a candidate
                            // set, neither of which can be sampled again.
                            if self.pending_forks.contains_key(id) {
                                need.full = true;
                                need.gumbel = None;
                            }
                            need
                        })
                        .collect();
                    Some(self.backend.forward_candidates(&plan.batch, &needs)?)
                }
            };
            let t_forward = t1.elapsed();
            let t2 = std::time::Instant::now();

            // Shared by every constrained row in this step. A constraint
            // that has not taken effect yet needs no mask.
            let masker = plan
                .sampled_seqs
                .iter()
                .any(|id| {
                    self.streams.get(id).is_some_and(|st| {
                        st.constraint
                            .as_ref()
                            .and_then(Constraint::machine)
                            .is_some()
                    })
                })
                .then(|| self.masker());
            let eos_ids = self.eos_ids();

            // (leader, extra choices, its logits row, its parameters).
            let mut forks_to_make: Vec<(SeqId, usize, Vec<f32>, vapi_core::SamplingParams)> =
                Vec::new();
            let mut sampled = Vec::with_capacity(plan.sampled_seqs.len());
            // Per row: (logprob of the sampled token, top alternatives),
            // for requests that asked.
            let mut logprobs: Vec<Option<(f32, Vec<TokenLogprob>)>> =
                Vec::with_capacity(plan.sampled_seqs.len());
            for (row, seq_id) in plan.sampled_seqs.iter().enumerate() {
                let params = self
                    .scheduler
                    .get(*seq_id)
                    .map(|s| s.params.clone())
                    .unwrap_or_default();
                // Cloned so the sampler can hold its own mutable borrow of
                // the stream state; a machine is a short stack and a shared
                // schema, so this is cheap.
                // Three quarters of the budget for thinking, the rest for
                // the document. Cutting thinking short costs answer
                // quality, so this fires late; it exists so a request
                // always comes back with something that parses.
                let out_of_patience = self
                    .scheduler
                    .get(*seq_id)
                    .is_some_and(|s| s.num_generated() * 4 >= s.params.max_tokens * 3);
                // Spend the last of the budget closing the document
                // rather than being cut off mid-value.
                let finishing = self.scheduler.get(*seq_id).is_some_and(|s| {
                    s.params.max_tokens.saturating_sub(s.num_generated()) <= CLOSING_BUDGET
                });
                let constraint = self
                    .streams
                    .get(seq_id)
                    .and_then(|st| st.constraint.as_ref())
                    .map(|c| c.row(out_of_patience))
                    .map(|c| match c {
                        RowConstraint::Active { machine, .. } => {
                            RowConstraint::Active { machine, finishing }
                        }
                        other => other,
                    });
                let state = &mut self
                    .streams
                    .get_mut(seq_id)
                    .expect("every scheduled sequence was admitted")
                    .sampler;
                // The full row, when the host has it: logprobs are read
                // off it before the sampler applies penalties in place.
                let mut full_row: Option<&mut [f32]> = match (&device_sampled, &mut logits) {
                    (None, Some(StepLogits::Full(l))) => Some(l.row_mut(row)),
                    (None, Some(StepLogits::Rows(rows))) => match &mut rows[row] {
                        RowLogits::Full(v) => Some(v.as_mut_slice()),
                        _ => None,
                    },
                    _ => None,
                };
                // Constrained decoding: rule out every token that would
                // break the format before anything is sampled. The row is
                // always the full one here, because `SamplerState::need`
                // asks for it whenever a constraint is set.
                match (&constraint, full_row.as_deref_mut()) {
                    (Some(RowConstraint::Active { machine, finishing }), Some(row_logits)) => {
                        if let Some(masker) = &masker {
                            if *finishing {
                                masker.mask_finishing(machine, &eos_ids, row_logits);
                            } else {
                                masker.mask(machine, &eos_ids, row_logits);
                            }
                        }
                    }
                    (
                        Some(RowConstraint::Waiting {
                            close_token,
                            out_of_patience,
                        }),
                        Some(row_logits),
                    ) => {
                        let wants_to_stop = *out_of_patience
                            || argmax(row_logits).is_some_and(|t| eos_ids.contains(&t));
                        match (wants_to_stop, close_token) {
                            // Done thinking: close the block, which is what
                            // puts the constraint in force.
                            (true, Some(t)) => {
                                let keep = *t as usize;
                                for (i, l) in row_logits.iter_mut().enumerate() {
                                    if i != keep {
                                        *l = f32::NEG_INFINITY;
                                    }
                                }
                            }
                            // No single token for the marker: the best that
                            // can be done is to refuse the end of turn.
                            _ => {
                                for e in &eos_ids {
                                    if let Some(l) = row_logits.get_mut(*e as usize) {
                                        *l = f32::NEG_INFINITY;
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
                // Kept for the fork path: the row as the model produced it.
                let full_row_snapshot: Option<Vec<f32>> = self
                    .pending_forks
                    .contains_key(seq_id)
                    .then(|| full_row.as_deref().map(<[f32]>::to_vec))
                    .flatten();
                let (raw, lp) = match (params.logprobs, full_row.map(|r| &*r)) {
                    (Some(k), Some(row)) => {
                        let lp = RowLogprobs::of(row, k);
                        (Some(row.to_vec()), Some(lp))
                    }
                    _ => (None, None),
                };
                let token = match (&device_sampled, &mut logits) {
                    (Some(tokens), _) => tokens[row],
                    (None, Some(StepLogits::Full(logits))) => {
                        Sampler::sample(logits.row_mut(row), &params, state)
                    }
                    (None, Some(StepLogits::Rows(rows))) => match &mut rows[row] {
                        RowLogits::Full(v) => Sampler::sample(v, &params, state),
                        RowLogits::Candidates(c) => Sampler::sample_candidates(c, &params, state),
                        RowLogits::Token(t) => {
                            state.observe_device_draw(*t);
                            sampled.push(*t);
                            logprobs.push(None);
                            continue;
                        }
                    },
                    (None, None) => unreachable!("one of the two forwards ran"),
                };
                // The first token of a request that wants several choices:
                // its siblings are sampled from this same row, since they
                // share the whole prompt. Snapshot it before the sampler's
                // penalties touch it again.
                if let Some(extra) = self.pending_forks.get(seq_id).copied() {
                    match full_row_snapshot.as_ref() {
                        Some(row) => {
                            forks_to_make.push((*seq_id, extra, row.clone(), params.clone()))
                        }
                        // Should not happen: the need above asks for the
                        // full row. Serve one choice rather than leave the
                        // caller waiting for completions that never come.
                        None => {
                            tracing::warn!("no logits to fork from; serving one choice");
                            forks_to_make.push((*seq_id, 0, Vec::new(), params.clone()));
                        }
                    }
                }
                state.observe(token);
                // Advance the constraint with what was actually sampled.
                // A token that does not fit cannot be sampled (it was
                // masked), so a rejection here would be a bug in the mask;
                // drop the constraint rather than fail the request.
                if let Some(st) = self.streams.get_mut(seq_id)
                    && let Some(constraint) = &mut st.constraint
                {
                    let text = self.tokenizer.decode(&[token]);
                    if !constraint.push(&text) {
                        tracing::warn!(
                            token,
                            text = %text,
                            "a sampled token broke the response format; dropping the constraint"
                        );
                        metrics::counter!("vapi_constraint_violations_total").increment(1);
                        st.constraint = None;
                    }
                }
                sampled.push(token);
                logprobs.push(match (raw, lp) {
                    (Some(raw), Some(lp)) => Some((
                        lp.logprob(raw[token as usize]),
                        lp.top
                            .iter()
                            .map(|&(token_id, logprob)| TokenLogprob { token_id, logprob })
                            .collect(),
                    )),
                    _ => None,
                });
            }

            // Emit text before committing, so a token that turns out to be EOS
            // is not streamed to the client as content.
            let eos = self.eos_ids();
            let mut stopped = Vec::new();
            for (row, (seq_id, &token)) in plan.sampled_seqs.iter().zip(&sampled).enumerate() {
                if eos.contains(&token) {
                    continue;
                }
                let Some(rid) = self.seq_to_request.get(seq_id).cloned() else {
                    continue;
                };
                let stop = self
                    .scheduler
                    .get(*seq_id)
                    .map(|s| s.params.stop.clone())
                    .unwrap_or_default();
                let Some(st) = self.streams.get_mut(seq_id) else {
                    continue;
                };
                let tokenizer = &self.tokenizer;
                let text = st.detok.push(token, |ids| tokenizer.decode(ids));
                st.last_token = token;

                if let Some(text) = emit_with_stop_strings(st, &stop, text) {
                    let (logprob, top_logprobs) = match logprobs.get_mut(row).and_then(Option::take)
                    {
                        Some((lp, top)) => (Some(lp), Some(top)),
                        None => (None, None),
                    };
                    out.events.push((
                        rid,
                        Delta::Token {
                            choice: self.choice_of.get(seq_id).copied().unwrap_or(0),
                            text,
                            token_id: token,
                            logprob,
                            top_logprobs,
                        },
                    ));
                }
                if st.stopped {
                    stopped.push(*seq_id);
                }
            }
            // Before commit, so a stop string that lands on the same token as
            // `max_tokens` is reported as "stop", which is what it was.
            for id in stopped {
                self.scheduler.finish_with(id, FinishReason::Stop);
            }

            let t_sample = t2.elapsed();
            let t3 = std::time::Instant::now();
            self.scheduler.commit(&plan, &sampled);
            for (leader, extra, row, params) in forks_to_make {
                if extra == 0 {
                    if let Some(rid) = self.seq_to_request.get(&leader).cloned() {
                        self.open_choices.insert(rid, 1);
                    }
                    self.pending_forks.remove(&leader);
                    continue;
                }
                if let Err(e) = self.fork_request(leader, extra, &row, &params, &mut out) {
                    tracing::warn!(error = %e, "could not fork for n > 1; serving one choice");
                    metrics::counter!("vapi_fork_failures_total").increment(1);
                    if let Some(rid) = self.seq_to_request.get(&leader).cloned() {
                        // The caller is waiting for a completion per choice,
                        // so close the ones that will never run rather than
                        // leaving the request to time out.
                        for choice in 1..=extra as u32 {
                            out.events.push((
                                rid.clone(),
                                Delta::Done {
                                    choice,
                                    reason: FinishReason::Error,
                                    prompt_tokens: 0,
                                    completion_tokens: 0,
                                },
                            ));
                        }
                        self.open_choices.insert(rid, 1);
                    }
                }
                self.pending_forks.remove(&leader);
            }
            let t_commit = t3.elapsed();
            self.record_step_timing(&plan, t_schedule, t_forward, t_sample, t_commit);
        }

        for seq_id in &plan.preempted {
            tracing::debug!(?seq_id, "preempted for lack of blocks");
        }

        for (seq_id, seq) in self.scheduler.drain_finished() {
            let reason = match seq.status {
                SeqStatus::Finished(r) => r,
                _ => FinishReason::Error,
            };
            let stream = self.streams.remove(&seq_id);
            let choice = self.choice_of.remove(&seq_id).unwrap_or(0);
            self.pending_forks.remove(&seq_id);
            if let Some(rid) = self.seq_to_request.remove(&seq_id) {
                // Text held back for a stop string that never came is still
                // the model's output; release it before the terminal event.
                if let Some(st) = stream
                    && !st.held.is_empty()
                    && reason != FinishReason::Cancelled
                {
                    out.events.push((
                        rid.clone(),
                        Delta::Token {
                            choice,
                            text: st.held,
                            token_id: st.last_token,
                            logprob: None,
                            top_logprobs: None,
                        },
                    ));
                }
                out.events.push((
                    rid.clone(),
                    Delta::Done {
                        choice,
                        reason,
                        prompt_tokens: seq.prompt_len(),
                        completion_tokens: seq.num_generated(),
                    },
                ));
                // The JetStream message is acked once, when the last choice
                // of the request is done.
                let left = self
                    .open_choices
                    .get_mut(&rid)
                    .map(|n| {
                        *n = n.saturating_sub(1);
                        *n
                    })
                    .unwrap_or(0);
                if left == 0 {
                    self.open_choices.remove(&rid);
                    out.completed.push(rid);
                }
            }
            out.did_work = true;
        }

        Ok(out)
    }

    /// Per-phase step timings as histograms, plus a periodic debug line so a
    /// load test can be read without a metrics scraper. Where a step's time
    /// goes is what decides the next optimisation, so this stays on.
    fn record_step_timing(
        &mut self,
        plan: &vapi_engine::StepPlan,
        schedule: std::time::Duration,
        forward: std::time::Duration,
        sample: std::time::Duration,
        commit: std::time::Duration,
    ) {
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        metrics::histogram!("vapi_step_schedule_ms").record(ms(schedule));
        metrics::histogram!("vapi_step_forward_ms").record(ms(forward));
        metrics::histogram!("vapi_step_sample_ms").record(ms(sample));
        metrics::histogram!("vapi_step_commit_ms").record(ms(commit));
        metrics::histogram!("vapi_step_batch_seqs").record(plan.batch.batch_size() as f64);
        metrics::histogram!("vapi_step_batch_tokens").record(plan.batch.num_tokens() as f64);
        self.steps += 1;
        if let Some(s) = self.spill_stats() {
            metrics::gauge!("vapi_spill_ram_bytes").set(s.ram_bytes as f64);
            metrics::gauge!("vapi_spill_disk_bytes").set(s.disk_bytes as f64);
            metrics::counter!("vapi_spill_hits_total").absolute(s.hits);
            metrics::counter!("vapi_spill_misses_total").absolute(s.misses);
            metrics::counter!("vapi_spill_writes_total").absolute(s.writes);
        }
        if self.steps.is_multiple_of(100) {
            tracing::debug!(
                seqs = plan.batch.batch_size(),
                tokens = plan.batch.num_tokens(),
                schedule_ms = format_args!("{:.2}", ms(schedule)),
                forward_ms = format_args!("{:.2}", ms(forward)),
                sample_ms = format_args!("{:.2}", ms(sample)),
                commit_ms = format_args!("{:.2}", ms(commit)),
                "step timing"
            );
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use vapi_core::{ModelId, SamplingParams};
    use vapi_engine::MockBackend;
    use vapi_proto::JobKind;

    fn engine(script: impl IntoIterator<Item = u32>) -> Engine {
        let mut cfg = Config::default();
        cfg.model.num_blocks = Some(16);
        cfg.model.max_context = 256;
        let tokenizer = Tokenization::bytes().unwrap();
        let mut spec = vapi_engine::ModelSpec::tiny();
        spec.vocab_size = 257;
        spec.eos_token_ids = tokenizer.eos_token_ids();
        let backend = MockBackend::new(16, vapi_cache::BLOCK_SIZE)
            .with_spec(spec)
            .with_script(script);
        Engine::new(&cfg, Box::new(backend), tokenizer, "w-test".into())
    }

    fn job(stop: &[&str], max_tokens: usize) -> Job {
        Job {
            request_id: RequestId::new(),
            model: ModelId("m".into()),
            kind: JobKind::Chat,
            prompt_tokens: b"hi".iter().map(|&b| b as u32).collect(),
            params: SamplingParams {
                temperature: 0.0,
                max_tokens,
                stop: StopCondition {
                    stop_strings: stop.iter().map(|s| s.to_string()).collect(),
                    ..Default::default()
                },
                ..Default::default()
            },
            namespace: "global".into(),
            reply_to: "x".into(),
            enqueued_at_ms: 0,
            rows: Vec::new(),
        }
    }

    /// A prompt of `n` distinct tokens, seeded by `tag` so two prompts
    /// share no block.
    fn long_prompt(tag: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| 1 + tag * 1000 + i).collect()
    }

    fn spill_engine(dir: &std::path::Path, num_blocks: usize) -> Engine {
        let mut cfg = Config::default();
        cfg.model.num_blocks = Some(num_blocks);
        cfg.model.max_context = 512;
        cfg.cache.spill = true;
        cfg.cache.spill_dir = dir.to_path_buf();
        cfg.cache.spill_ram_bytes = 1 << 20;
        cfg.cache.spill_max_bytes = 1 << 20;
        cfg.worker.max_concurrent_seqs = 4;
        let tokenizer = Tokenization::bytes().unwrap();
        let mut spec = vapi_engine::ModelSpec::tiny();
        spec.vocab_size = 100_000;
        spec.eos_token_ids = vec![];
        let backend = MockBackend::new(num_blocks, vapi_cache::BLOCK_SIZE)
            .with_spec(spec)
            .with_script([7u32])
            .with_movable_blocks(64);
        Engine::new(&cfg, Box::new(backend), tokenizer, "w-spill".into())
    }

    /// Run one prompt to completion and report the cached prefix the worker
    /// announced for it.
    fn run_prompt(engine: &mut Engine, prompt: Vec<u32>) -> usize {
        // More than one token: a sequence that finishes on the same step it
        // finishes prefilling has already given its blocks back by the time
        // the scheduler publishes, so it would never populate the cache.
        let mut j = job(&[], 3);
        j.prompt_tokens = prompt;
        engine.admit(j).unwrap();
        let mut cached = 0;
        for _ in 0..50 {
            let out = engine.step().unwrap();
            for (_, d) in &out.events {
                if let Delta::Started {
                    cached_prefix_tokens,
                    ..
                } = d
                {
                    cached = *cached_prefix_tokens;
                }
            }
            if engine.is_idle() {
                break;
            }
        }
        cached
    }

    #[test]
    fn evicted_blocks_spill_and_come_back_on_the_next_request() {
        let dir = std::env::temp_dir().join(format!("vapi-spill-engine-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let block = vapi_cache::BLOCK_SIZE;
        let mut engine = spill_engine(&dir, 4);

        // First pass: nothing is cached anywhere.
        let a = long_prompt(1, 2 * block);
        assert_eq!(run_prompt(&mut engine, a.clone()), 0);
        assert_eq!(
            engine.spill_stats().unwrap().writes,
            0,
            "nothing evicted yet"
        );

        // Other prompts push A's blocks out of the pool; their content goes
        // to the spill tier instead of being lost.
        run_prompt(&mut engine, long_prompt(2, 2 * block));
        run_prompt(&mut engine, long_prompt(3, 2 * block));
        let st = engine.spill_stats().unwrap();
        assert!(st.writes >= 2, "{st:?}");

        // A again: the prefix comes back from the spill tier, so the
        // scheduler sees it as a cache hit rather than prefilling it.
        let cached = run_prompt(&mut engine, a);
        let st = engine.spill_stats().unwrap();
        assert!(st.hits >= 2, "both blocks came back: {st:?}");
        // A fully cached prompt still recomputes its last block, because the
        // forward has to produce logits from somewhere; the block before it
        // is the saving.
        assert_eq!(cached, block, "the restored prefix was reused");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn restored_blocks_carry_the_original_kv_bytes() {
        // The bytes that come back must be the ones that went out: a
        // restored block that holds anything else is silent corruption.
        let dir = std::env::temp_dir().join(format!("vapi-spill-bytes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = spill_engine(&dir, 4);
        let id = vapi_cache::BlockId(2);
        let bytes: Vec<u8> = (0..64).map(|i| (i * 3) as u8).collect();
        engine.backend.import_block(id, &bytes).unwrap();
        assert_eq!(engine.backend.export_block(id).unwrap(), bytes);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_spill_tier_stays_off_without_a_backend_that_can_move_blocks() {
        let mut cfg = Config::default();
        cfg.model.num_blocks = Some(8);
        cfg.cache.spill = true;
        let tokenizer = Tokenization::bytes().unwrap();
        let backend = MockBackend::new(8, vapi_cache::BLOCK_SIZE);
        let engine = Engine::new(&cfg, Box::new(backend), tokenizer, "w".into());
        assert!(engine.spill_stats().is_none());
    }

    #[test]
    fn a_response_format_constrains_what_can_be_sampled() {
        // The mock's one-hot logits would sample token 7 every step; with a
        // constraint in force, only tokens that keep the document valid can
        // be chosen, so the output has to parse.
        let mut cfg = Config::default();
        cfg.model.num_blocks = Some(16);
        cfg.model.max_context = 256;
        let tokenizer = Tokenization::bytes().unwrap();
        let mut spec = vapi_engine::ModelSpec::tiny();
        spec.vocab_size = 257;
        spec.eos_token_ids = tokenizer.eos_token_ids();
        // A script that would produce "xx..." without a constraint. Every
        // other token has the same logit, so the sampler takes the lowest
        // id the mask leaves, which is a tab: the output is valid JSON
        // padded with the most whitespace the machine allows. A real model
        // does not do this, and the point here is that nothing invalid can
        // be sampled however the logits fall.
        let backend = MockBackend::new(16, vapi_cache::BLOCK_SIZE)
            .with_spec(spec)
            .with_script([b'x' as u32]);
        let mut engine = Engine::new(&cfg, Box::new(backend), tokenizer, "w-json".into());

        let mut j = job(&[], 128);
        j.params.response_format = Some(ResponseFormat::JsonSchema {
            schema: serde_json::json!({
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            }),
        });
        engine.admit(j).unwrap();
        let (text, reason, _) = drive(&mut engine);
        assert!(
            serde_json::from_str::<serde_json::Value>(&text).is_ok(),
            "output must parse: {text:?} ({reason:?})"
        );
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(v["ok"].is_boolean(), "{text}");
        assert!(!text.contains('x'), "the scripted token was masked: {text}");
        assert_eq!(
            reason,
            FinishReason::Stop,
            "it stopped at the closing brace"
        );
    }

    #[test]
    fn an_unsupported_schema_is_refused_at_admission() {
        let mut engine = engine([b'a' as u32]);
        let mut j = job(&[], 4);
        j.params.response_format = Some(ResponseFormat::JsonSchema {
            schema: serde_json::json!({"type": "string", "pattern": "^a"}),
        });
        let err = engine.admit(j).unwrap_err().to_string();
        assert!(err.contains("pattern"), "{err}");
        assert!(engine.is_idle(), "nothing was admitted");
    }

    #[test]
    fn n_greater_than_one_forks_and_reports_every_choice() {
        // The mock scripts one token, so every choice generates the same
        // text; what is under test is that three of them exist, that they
        // each stream under their own index, and that the request is
        // reported complete once, after the last one.
        let mut engine = engine([b'a' as u32, b'b' as u32]);
        let mut j = job(&[], 4);
        j.params.n = 3;
        j.params.temperature = 0.7;
        let rid = j.request_id.clone();
        engine.admit(j).unwrap();

        let mut per_choice: std::collections::HashMap<u32, String> = Default::default();
        let mut dones = Vec::new();
        let mut completed = 0;
        for _ in 0..80 {
            let out = engine.step().unwrap();
            for (r, d) in &out.events {
                assert_eq!(*r, rid);
                match d {
                    Delta::Token { choice, text, .. } => {
                        per_choice.entry(*choice).or_default().push_str(text);
                    }
                    Delta::Done { choice, .. } => dones.push(*choice),
                    _ => {}
                }
            }
            completed += out.completed.len();
            if engine.is_idle() {
                break;
            }
        }
        assert_eq!(per_choice.len(), 3, "three choices: {per_choice:?}");
        assert!(
            per_choice.values().all(|t| t.len() == 4),
            "each ran to max_tokens: {per_choice:?}"
        );
        dones.sort_unstable();
        assert_eq!(dones, vec![0, 1, 2]);
        assert_eq!(completed, 1, "one JetStream ack for the whole request");
        assert_eq!(
            engine.scheduler.pool().num_free(),
            16,
            "every block came back"
        );
    }

    #[test]
    fn a_greedy_request_is_not_forked() {
        // Greedy choices would be identical, so asking for several is
        // served by one sequence rather than n copies of the same answer.
        let mut engine = engine([b'a' as u32]);
        let mut j = job(&[], 3);
        j.params.n = 4;
        engine.admit(j).unwrap();
        let (text, _, _) = drive(&mut engine);
        assert_eq!(text, "aaa");
    }

    #[test]
    fn logprobs_are_reported_when_asked() {
        let mut eng = engine([b'a' as u32, b'b' as u32]);
        let mut j = job(&[], 2);
        j.params.logprobs = Some(2);
        eng.admit(j).unwrap();
        let mut seen = 0;
        for _ in 0..10 {
            let out = eng.step().unwrap();
            for (_, d) in out.events {
                if let Delta::Token {
                    choice: 0,
                    logprob,
                    top_logprobs,
                    token_id,
                    ..
                } = d
                {
                    seen += 1;
                    // The mock's one-hot row: the scripted token has logit
                    // 100, everything else 0, so its logprob is ~0.
                    let lp = logprob.expect("logprob present");
                    assert!(lp > -1e-3 && lp <= 0.0, "{lp}");
                    let top = top_logprobs.expect("top logprobs present");
                    assert_eq!(top.len(), 2);
                    assert_eq!(top[0].token_id, token_id);
                    assert!(top[1].logprob < -90.0);
                }
            }
            if eng.is_idle() {
                break;
            }
        }
        assert_eq!(seen, 2);
        // Not asked: nothing reported.
        let mut plain = engine([b'a' as u32]);
        plain.admit(job(&[], 1)).unwrap();
        let out = plain.step().unwrap();
        assert!(out.events.iter().all(|(_, d)| !matches!(
            d,
            Delta::Token {
                choice: 0,
                logprob: Some(_),
                ..
            }
        )));
    }

    #[test]
    fn a_failed_step_fails_every_request_and_frees_the_engine() {
        let mut cfg = Config::default();
        cfg.model.num_blocks = Some(16);
        cfg.model.max_context = 256;
        let tokenizer = Tokenization::bytes().unwrap();
        let mut spec = vapi_engine::ModelSpec::tiny();
        spec.vocab_size = 257;
        spec.eos_token_ids = tokenizer.eos_token_ids();
        let backend = MockBackend::new(16, vapi_cache::BLOCK_SIZE)
            .with_spec(spec)
            .with_script([b'a' as u32])
            .failing_at(2);
        let mut engine = Engine::new(&cfg, Box::new(backend), tokenizer, "w-test".into());
        let a = job(&[], 50);
        let b = job(&[], 50);
        let (ra, rb) = (a.request_id.clone(), b.request_id.clone());
        engine.admit(a).unwrap();
        engine.admit(b).unwrap();
        engine.step().unwrap();
        engine.step().unwrap();
        let err = engine.step().expect_err("the third step fails");
        let out = engine.fail_all(&err.to_string());
        let failed: Vec<_> = out
            .events
            .iter()
            .filter_map(|(rid, d)| match d {
                Delta::Failed { message } => Some((rid.clone(), message.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(failed.len(), 2);
        assert!(failed.iter().all(|(_, m)| m.contains("injected")));
        assert!(failed.iter().any(|(r, _)| *r == ra) && failed.iter().any(|(r, _)| *r == rb));
        assert_eq!(out.completed.len(), 2);
        assert!(engine.is_idle());
        assert_eq!(engine.headroom(), cfg.worker.max_concurrent_seqs);
        assert_eq!(
            engine.scheduler.pool().num_free(),
            16,
            "every block is back"
        );
        // And it still serves.
        let c = job(&[], 3);
        engine.admit(c).unwrap();
        let (text, reason, _) = drive(&mut engine);
        assert_eq!(text, "aaa");
        assert_eq!(reason, FinishReason::Length);
    }

    /// Run to completion, returning the streamed text and the finish reason.
    fn drive(engine: &mut Engine) -> (String, FinishReason, usize) {
        let mut text = String::new();
        for _ in 0..200 {
            let out = engine.step().unwrap();
            for (_, d) in out.events {
                match d {
                    Delta::Token {
                        choice: 0, text: t, ..
                    } => text.push_str(&t),
                    Delta::Done {
                        choice: 0,
                        reason,
                        completion_tokens,
                        ..
                    } => return (text, reason, completion_tokens),
                    _ => {}
                }
            }
        }
        panic!("never finished");
    }

    fn bytes(s: &str) -> Vec<u32> {
        s.bytes().map(u32::from).collect()
    }

    #[test]
    fn a_stop_string_truncates_the_output_and_finishes_with_stop() {
        // F4: `stop` used to be parsed and then ignored.
        let mut script = bytes("ab\nUser: never seen");
        script.push(vapi_tokenize::BYTE_EOS);
        let mut e = engine(script);
        e.admit(job(&["\nUser:"], 100)).unwrap();
        let (text, reason, completion_tokens) = drive(&mut e);
        assert_eq!(text, "ab");
        assert_eq!(reason, FinishReason::Stop);
        // a b \n U s e r : — the token that completed the stop string counts.
        assert_eq!(completion_tokens, 8);
    }

    #[test]
    fn held_back_text_is_released_when_no_stop_string_arrives() {
        let mut script = bytes("abc");
        script.push(vapi_tokenize::BYTE_EOS);
        let mut e = engine(script);
        e.admit(job(&["zzz"], 100)).unwrap();
        let (text, reason, _) = drive(&mut e);
        assert_eq!(text, "abc", "nothing may be swallowed by the hold-back");
        assert_eq!(reason, FinishReason::Stop);
    }

    #[test]
    fn a_stop_string_split_across_tokens_is_still_caught() {
        // Each byte is its own token, so every stop string spans tokens.
        let mut e = engine(bytes("xxENDyy"));
        e.admit(job(&["END"], 100)).unwrap();
        let (text, reason, _) = drive(&mut e);
        assert_eq!(text, "xx");
        assert_eq!(reason, FinishReason::Stop);
    }

    #[test]
    fn without_stop_strings_text_streams_immediately() {
        let mut e = engine(bytes("abcdef"));
        e.admit(job(&[], 3)).unwrap();
        let out = e.step().unwrap();
        let out2 = e.step().unwrap();
        let first: Vec<_> = out
            .events
            .iter()
            .chain(out2.events.iter())
            .filter_map(|(_, d)| match d {
                Delta::Token {
                    choice: 0, text, ..
                } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(first, vec!["a", "b"], "one token out per step, unbuffered");
        let (text, reason, _) = drive(&mut e);
        assert_eq!(text, "c");
        assert_eq!(reason, FinishReason::Length);
    }
}
