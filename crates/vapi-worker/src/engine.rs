use std::collections::HashMap;

use vapi_cache::{BlockPool, CacheNamespace, SpillStore, hash_block_chain};
use vapi_core::{Config, FinishReason, RequestId, Result, SeqId, StopCondition};
use vapi_engine::backend::ExecutionBackend;
use vapi_engine::{
    CandidateNeed, IncrementalDetokenizer, RowLogits, RowLogprobs, Sampler, SamplerState,
    Scheduler, SchedulerConfig, SeqStatus, StepLogits,
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
    delta_seq: HashMap<RequestId, DeltaSeq>,
    worker_id: String,
    steps: u64,
    /// Tier 3. `None` when it is off or the backend cannot move blocks.
    spill: Option<SpillStore>,
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
            delta_seq: HashMap::new(),
            worker_id,
            steps: 0,
            spill,
            namespace,
        }
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
        let prompt = job.prompt_tokens;
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
            },
        );
        self.seq_to_request.insert(id, rid.clone());
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
                            self.streams
                                .get(id)
                                .map(|st| st.sampler.need(&params))
                                .unwrap_or(CandidateNeed {
                                    inv_temperature: 1.0,
                                    needed: 0,
                                    gumbel: None,
                                    full: false,
                                })
                        })
                        .collect();
                    Some(self.backend.forward_candidates(&plan.batch, &needs)?)
                }
            };
            let t_forward = t1.elapsed();
            let t2 = std::time::Instant::now();

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
                let state = &mut self
                    .streams
                    .get_mut(seq_id)
                    .expect("every scheduled sequence was admitted")
                    .sampler;
                // The full row, when the host has it: logprobs are read
                // off it before the sampler applies penalties in place.
                let full_row: Option<&mut [f32]> = match (&device_sampled, &mut logits) {
                    (None, Some(StepLogits::Full(l))) => Some(l.row_mut(row)),
                    (None, Some(StepLogits::Rows(rows))) => match &mut rows[row] {
                        RowLogits::Full(v) => Some(v.as_mut_slice()),
                        _ => None,
                    },
                    _ => None,
                };
                let (raw, lp) = match (params.logprobs, full_row) {
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
                state.observe(token);
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
                        reason,
                        prompt_tokens: seq.prompt_len(),
                        completion_tokens: seq.num_generated(),
                    },
                ));
                out.completed.push(rid);
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
                    Delta::Token { text: t, .. } => text.push_str(&t),
                    Delta::Done {
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
                Delta::Token { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(first, vec!["a", "b"], "one token out per step, unbuffered");
        let (text, reason, _) = drive(&mut e);
        assert_eq!(text, "c");
        assert_eq!(reason, FinishReason::Length);
    }
}
