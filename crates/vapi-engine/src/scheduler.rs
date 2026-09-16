use std::collections::{HashMap, VecDeque};

use vapi_cache::{BlockPool, CacheNamespace, hash_block_chain};
use vapi_core::{FinishReason, RequestId, Result, SamplingParams, SeqId};

use crate::backend::ForwardBatch;
use crate::sequence::{SeqStatus, Sequence};

#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub max_concurrent_seqs: usize,
    pub max_batched_tokens: usize,
    pub prefill_chunk_tokens: usize,
    pub max_context: usize,
    pub block_size: usize,
    pub prefix_cache: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_seqs: 64,
            max_batched_tokens: 8192,
            prefill_chunk_tokens: 2048,
            max_context: 8192,
            block_size: vapi_cache::BLOCK_SIZE,
            prefix_cache: true,
        }
    }
}

/// One step's worth of work, plus what the scheduler decided along the way.
#[derive(Debug, Default)]
pub struct StepPlan {
    pub batch: ForwardBatch,
    /// Sequence occupying each batch slot, in `cu_seqlens` order. Recorded as
    /// the batch is built rather than reconstructed afterwards — the slot to
    /// sequence mapping is not recoverable from the tensors alone.
    pub batch_seqs: Vec<SeqId>,
    /// Sequence behind each entry of `batch.logits_indices`, in the same
    /// order, so sampled tokens can be routed back.
    pub sampled_seqs: Vec<SeqId>,
    /// Sequences evicted to make room this step.
    pub preempted: Vec<SeqId>,
    /// Sequences that began running this step, with how many prompt tokens
    /// the shared cache supplied.
    pub started: Vec<(SeqId, usize)>,
}

impl StepPlan {
    pub fn is_empty(&self) -> bool {
        self.batch.is_empty()
    }
}

/// Continuous-batching scheduler over a shared block pool.
///
/// Every sequence is owned by exactly one `HashMap`; the queues hold ids only.
/// That discipline is deliberate — the characteristic scheduler bugs are
/// "a sequence is in two queues at once" and "a cancelled sequence was
/// scheduled after being freed", and they are much easier to prevent than to
/// debug.
pub struct Scheduler {
    cfg: SchedulerConfig,
    pool: BlockPool,
    namespace: CacheNamespace,
    seqs: HashMap<SeqId, Sequence>,
    by_request: HashMap<RequestId, SeqId>,
    waiting: VecDeque<SeqId>,
    running: Vec<SeqId>,
    finished: Vec<SeqId>,
    next_id: u64,
    eos_token_ids: Vec<u32>,
}

impl Scheduler {
    pub fn new(
        cfg: SchedulerConfig,
        pool: BlockPool,
        namespace: CacheNamespace,
        eos_token_ids: Vec<u32>,
    ) -> Self {
        Self {
            cfg,
            pool,
            namespace,
            seqs: HashMap::new(),
            by_request: HashMap::new(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            finished: Vec::new(),
            next_id: 0,
            eos_token_ids,
        }
    }

    pub fn pool(&self) -> &BlockPool {
        &self.pool
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Whether any completed sequence is still waiting to be drained.
    ///
    /// The engine loop must keep stepping while this is true: a sequence
    /// cancelled or finished between steps has had its blocks freed but its
    /// terminal event not yet reported, and going idle here would strand it.
    pub fn has_finished(&self) -> bool {
        !self.finished.is_empty()
    }

    /// How many more sequences can be admitted. The worker feeds this to
    /// JetStream as its `max_ack_pending` credit, so NATS stops delivering
    /// work before the engine is oversubscribed.
    pub fn admission_headroom(&self) -> usize {
        self.cfg
            .max_concurrent_seqs
            .saturating_sub(self.running.len() + self.waiting.len())
    }

    pub fn get(&self, id: SeqId) -> Option<&Sequence> {
        self.seqs.get(&id)
    }

    /// Queue a request. Rejects prompts that cannot fit the context window
    /// here rather than discovering it mid-prefill.
    pub fn admit(
        &mut self,
        request_id: RequestId,
        prompt: Vec<u32>,
        params: SamplingParams,
        namespace: String,
    ) -> Result<SeqId> {
        if prompt.is_empty() {
            return Err(vapi_core::Error::InvalidRequest("empty prompt".into()));
        }
        if prompt.len() >= self.cfg.max_context {
            return Err(vapi_core::Error::ContextLengthExceeded {
                tokens: prompt.len(),
                limit: self.cfg.max_context,
            });
        }
        let id = SeqId(self.next_id);
        self.next_id += 1;
        self.by_request.insert(request_id.clone(), id);
        self.seqs
            .insert(id, Sequence::new(request_id, prompt, params, namespace));
        self.waiting.push_back(id);
        Ok(id)
    }

    /// Abort a request, freeing its blocks at the next safe point.
    pub fn cancel(&mut self, request_id: &RequestId) -> Option<SeqId> {
        let id = *self.by_request.get(request_id)?;
        self.finish(id, FinishReason::Cancelled);
        Some(id)
    }

    /// Build the next batch.
    ///
    /// Decodes are scheduled before prefills. A decode step is one token for a
    /// sequence a user is actively watching, so making it wait behind a
    /// multi-thousand-token prefill is what turns a good p50 into a terrible
    /// p99. Prefills then fill whatever token budget is left, in chunks.
    pub fn schedule(&mut self) -> StepPlan {
        let mut plan = StepPlan::default();
        let mut budget = self.cfg.max_batched_tokens;

        self.schedule_decodes(&mut plan, &mut budget);
        self.schedule_prefills(&mut plan, &mut budget);

        plan.batch.max_seqlen_q = plan
            .batch
            .cu_seqlens_q
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0);
        plan.batch.max_seqlen_k = plan
            .batch
            .cu_seqlens_k
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0);
        plan
    }

    fn schedule_decodes(&mut self, plan: &mut StepPlan, budget: &mut usize) {
        let running = std::mem::take(&mut self.running);
        let mut survivors = Vec::with_capacity(running.len());

        for id in running {
            let Some(seq) = self.seqs.get(&id) else {
                continue;
            };
            if seq.status.is_finished() {
                continue;
            }
            if seq.needs_prefill() {
                // Mid-prefill; handled by the prefill pass.
                survivors.push(id);
                continue;
            }
            if *budget == 0 {
                survivors.push(id);
                continue;
            }
            match self.try_extend(id, 1) {
                Ok(()) => {
                    self.push_seq_into_batch(id, 1, plan);
                    *budget -= 1;
                    survivors.push(id);
                }
                Err(()) => {
                    // Out of blocks. Preempt by recompute: cheaper to build
                    // than swapping, and correct because the sequence keeps
                    // the tokens it already emitted.
                    self.preempt(id);
                    plan.preempted.push(id);
                }
            }
        }
        self.running = survivors;
    }

    fn schedule_prefills(&mut self, plan: &mut StepPlan, budget: &mut usize) {
        // Sequences already running but still prefilling continue first, so a
        // chunked prompt makes progress rather than being restarted.
        let running = self.running.clone();
        for id in running {
            if *budget == 0 {
                return;
            }
            let Some(seq) = self.seqs.get(&id) else {
                continue;
            };
            if !seq.needs_prefill() || seq.status.is_finished() {
                continue;
            }
            self.prefill_chunk(id, plan, budget);
        }

        while *budget > 0 && self.running.len() < self.cfg.max_concurrent_seqs {
            let Some(id) = self.waiting.pop_front() else {
                return;
            };
            let Some(seq) = self.seqs.get(&id) else {
                continue;
            };
            if seq.status.is_finished() {
                continue;
            }

            if !self.try_start(id) {
                // Not enough blocks to admit it; put it back and stop trying,
                // so a large request cannot be starved by smaller ones behind
                // it.
                self.waiting.push_front(id);
                return;
            }
            let cached = self.seqs[&id].cached_prefix_tokens;
            plan.started.push((id, cached));
            self.running.push(id);
            self.prefill_chunk(id, plan, budget);
        }
    }

    /// Reserve blocks for a newly admitted sequence, consulting the shared
    /// prefix cache first.
    fn try_start(&mut self, id: SeqId) -> bool {
        let (prompt_len, tokens, namespace_label) = {
            let s = &self.seqs[&id];
            (s.prompt_len(), s.tokens().to_vec(), s.namespace.clone())
        };
        let _ = namespace_label;

        let mut matched = Vec::new();
        let mut cached_tokens = 0usize;
        if self.cfg.prefix_cache {
            let hashes =
                hash_block_chain(&self.namespace, &tokens[..prompt_len], self.cfg.block_size);
            let m = self.pool.match_prefix(&hashes);
            matched = m.blocks;
            cached_tokens = m.tokens;

            // A fully cached prompt still needs one token forwarded to produce
            // logits, so give back the final block and recompute it. Without
            // this carve-out a 100%-hit request has nothing to sample from.
            while cached_tokens >= prompt_len && !matched.is_empty() {
                let last = matched.pop().expect("non-empty");
                self.pool.free(last);
                cached_tokens -= self.cfg.block_size;
            }
        }

        // Blocks needed to hold the whole prompt plus its first generated
        // token; the rest are taken lazily as generation proceeds.
        let need_total = self.pool.blocks_for(prompt_len + 1);
        let need_new = need_total.saturating_sub(matched.len());
        if !self.pool.can_allocate(need_new) {
            self.pool.free_all(&matched);
            return false;
        }

        let mut blocks = matched;
        let n_cached = blocks.len();
        for _ in 0..need_new {
            match self.pool.allocate() {
                Ok(b) => blocks.push(b),
                Err(_) => {
                    self.pool.free_all(&blocks);
                    return false;
                }
            }
        }

        let seq = self.seqs.get_mut(&id).expect("admitted");
        let cached_blocks: Vec<_> = blocks[..n_cached].to_vec();
        seq.adopt_cached_prefix(cached_blocks, cached_tokens);
        seq.blocks = blocks;
        seq.status = SeqStatus::Prefilling;
        true
    }

    /// Ensure the sequence has blocks covering `extra` more positions.
    fn try_extend(&mut self, id: SeqId, extra: usize) -> std::result::Result<(), ()> {
        let need = {
            let s = &self.seqs[&id];
            self.pool.blocks_for(s.num_computed() + extra)
        };
        let have = self.seqs[&id].blocks.len();
        for _ in have..need {
            match self.pool.allocate() {
                Ok(b) => self.seqs.get_mut(&id).expect("live").blocks.push(b),
                Err(_) => return Err(()),
            }
        }
        Ok(())
    }

    fn prefill_chunk(&mut self, id: SeqId, plan: &mut StepPlan, budget: &mut usize) {
        let remaining = self.seqs[&id].remaining_prefill();
        let q = remaining.min(self.cfg.prefill_chunk_tokens).min(*budget);
        if q == 0 {
            return;
        }
        if self.try_extend(id, q).is_err() {
            self.preempt(id);
            plan.preempted.push(id);
            return;
        }
        self.push_seq_into_batch(id, q, plan);
        *budget -= q;
    }

    /// Append one sequence's contribution to the batch.
    fn push_seq_into_batch(&mut self, id: SeqId, q: usize, plan: &mut StepPlan) {
        let seq = &self.seqs[&id];
        let start = seq.num_computed();
        let bs = self.cfg.block_size;

        if plan.batch.cu_seqlens_q.is_empty() {
            plan.batch.cu_seqlens_q.push(0);
            plan.batch.cu_seqlens_k.push(0);
        }

        for i in start..start + q {
            plan.batch.tokens.push(seq.tokens()[i]);
            plan.batch.positions.push(i as u32);
            let block = seq.blocks[i / bs];
            plan.batch
                .slot_mapping
                .push((block.0 as usize * bs + (i % bs)) as u32);
        }

        let prev_q = *plan.batch.cu_seqlens_q.last().expect("seeded");
        let prev_k = *plan.batch.cu_seqlens_k.last().expect("seeded");
        plan.batch.cu_seqlens_q.push(prev_q + q as u32);
        // Keys include everything cached plus what is being computed now.
        plan.batch.cu_seqlens_k.push(prev_k + (start + q) as u32);
        plan.batch
            .block_tables
            .push(seq.blocks.iter().map(|b| b.0).collect());
        plan.batch_seqs.push(id);

        // Logits are only needed where a token will actually be sampled: at
        // the end of a decode step, or at the end of the *final* prefill
        // chunk. Requesting them for every position of a 2048-token chunk
        // would return 2048 × vocab floats to throw away.
        let completes = start + q >= seq.total_len();
        if completes {
            plan.batch
                .logits_indices
                .push((plan.batch.tokens.len() - 1) as u32);
            plan.sampled_seqs.push(id);
        }
    }

    fn preempt(&mut self, id: SeqId) {
        let blocks = {
            let seq = self.seqs.get_mut(&id).expect("live");
            std::mem::take(&mut seq.blocks)
        };
        self.pool.free_all(&blocks);
        let seq = self.seqs.get_mut(&id).expect("live");
        seq.reset_for_recompute();
        self.running.retain(|&r| r != id);
        self.waiting.push_front(id);
    }

    /// Apply a step's results: advance progress, append sampled tokens, and
    /// retire anything that finished.
    pub fn commit(&mut self, plan: &StepPlan, sampled: &[u32]) {
        debug_assert_eq!(sampled.len(), plan.sampled_seqs.len());

        // Every sequence in the batch advances by however many tokens it
        // contributed, whether or not it produced a logit.
        for (i, &id) in plan.batch_seqs.iter().enumerate() {
            let q = (plan.batch.cu_seqlens_q[i + 1] - plan.batch.cu_seqlens_q[i]) as usize;
            if let Some(seq) = self.seqs.get_mut(&id) {
                seq.advance_computed(q);
                if !seq.needs_prefill() && seq.status == SeqStatus::Prefilling {
                    seq.status = SeqStatus::Decoding;
                }
            }
        }

        for (id, &token) in plan.sampled_seqs.iter().zip(sampled) {
            let Some(seq) = self.seqs.get_mut(id) else {
                continue;
            };
            seq.push_token(token);
            if let Some(reason) = seq.check_finished(&self.eos_token_ids, self.cfg.max_context) {
                self.finish(*id, reason);
            }
        }

        self.publish_full_blocks(plan);
    }

    /// Publish newly filled blocks so other requests can reuse them.
    ///
    /// Only blocks that are completely full and whose contents are settled get
    /// published — a partially filled block will receive more tokens, and
    /// publishing it would let another sequence match a prefix that does not
    /// exist yet.
    fn publish_full_blocks(&mut self, plan: &StepPlan) {
        if !self.cfg.prefix_cache {
            return;
        }
        let bs = self.cfg.block_size;
        for &id in &plan.batch_seqs {
            let Some(seq) = self.seqs.get(&id) else {
                continue;
            };
            let full_blocks = seq.num_computed() / bs;
            if full_blocks <= seq.num_cached_blocks {
                continue;
            }
            let hashes = hash_block_chain(&self.namespace, &seq.tokens()[..full_blocks * bs], bs);
            let mut swaps = Vec::new();
            for (i, &hash) in hashes
                .iter()
                .enumerate()
                .take(full_blocks)
                .skip(seq.num_cached_blocks)
            {
                let Some(&block) = seq.blocks.get(i) else {
                    continue;
                };
                let canonical = self.pool.publish(block, hash);
                if canonical != block {
                    swaps.push((i, block, canonical));
                }
            }
            let seq = self.seqs.get_mut(&id).expect("live");
            seq.num_cached_blocks = full_blocks;
            for (i, old, canonical) in swaps {
                // Another sequence published identical content first; adopt
                // its block and release ours.
                seq.blocks[i] = canonical;
                self.pool.free(old);
            }
        }
    }

    fn finish(&mut self, id: SeqId, reason: FinishReason) {
        let blocks = {
            let Some(seq) = self.seqs.get_mut(&id) else {
                return;
            };
            if seq.status.is_finished() {
                return;
            }
            seq.status = SeqStatus::Finished(reason);
            std::mem::take(&mut seq.blocks)
        };
        self.pool.free_all(&blocks);
        self.running.retain(|&r| r != id);
        self.waiting.retain(|&r| r != id);
        self.finished.push(id);
    }

    /// Remove and return everything that completed since the last call.
    pub fn drain_finished(&mut self) -> Vec<(SeqId, Sequence)> {
        let ids = std::mem::take(&mut self.finished);
        ids.into_iter()
            .filter_map(|id| {
                let seq = self.seqs.remove(&id)?;
                self.by_request.remove(&seq.request_id);
                Some((id, seq))
            })
            .collect()
    }
}
