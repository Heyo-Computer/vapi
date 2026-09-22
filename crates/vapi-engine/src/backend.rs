use std::collections::HashMap;

use vapi_cache::BlockId;
use vapi_core::Result;

/// Static description of a loaded model, enough for the engine to size the KV
/// cache and build batches without knowing anything about tensors.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelSpec {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_context: usize,
    pub eos_token_ids: Vec<u32>,
}

impl ModelSpec {
    /// Bytes of KV cache per token, across all layers, for K and V together.
    /// This is what turns a VRAM budget into a block count.
    pub fn kv_bytes_per_token(&self, dtype_bytes: usize) -> usize {
        2 * self.num_layers * self.num_kv_heads * self.head_dim * dtype_bytes
    }

    /// A deliberately tiny model for tests: small enough that block
    /// accounting can be checked by hand.
    pub fn tiny() -> Self {
        Self {
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 8,
            vocab_size: 256,
            max_context: 512,
            eos_token_ids: vec![0],
        }
    }
}

/// One scheduler step's work, in exactly the shape a paged attention kernel
/// wants.
///
/// The layout mirrors `flash_attn_varlen_paged_windowed`'s arguments on
/// purpose. Because that call is *varlen* and takes a block table, prefill and
/// decode are not separate code paths: a prefill chunk is a sequence
/// contributing many query tokens, a decode step is one contributing a single
/// token, and a mixed batch is simply both in one set of cumulative lengths.
#[derive(Clone, Debug, Default)]
pub struct ForwardBatch {
    /// Token ids for every sequence, concatenated.
    pub tokens: Vec<u32>,
    /// Absolute position of each token within its own sequence, which is what
    /// RoPE needs. Cannot be derived from the batch alone once sequences sit
    /// at different offsets — the reason the stock candle `forward(x, index_pos)`
    /// signature cannot express a continuous batch.
    pub positions: Vec<u32>,
    /// Cumulative query lengths, `batch_size + 1` entries starting at 0.
    pub cu_seqlens_q: Vec<u32>,
    /// Cumulative key lengths including each sequence's cached prefix,
    /// `batch_size + 1` entries. Larger than `cu_seqlens_q` exactly when a
    /// sequence is attending over tokens it is not recomputing.
    pub cu_seqlens_k: Vec<u32>,
    /// Flat KV cache slot each token's K/V must be written to:
    /// `block_table[pos / BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE`.
    pub slot_mapping: Vec<u32>,
    /// Per-sequence block tables. Packed into a `[batch, max_blocks]` device
    /// tensor by the CUDA backend.
    pub block_tables: Vec<Vec<u32>>,
    /// Indices into `tokens` whose logits the sampler needs — the last token
    /// of each sequence. Slicing here rather than returning logits for every
    /// position is what keeps a chunked prefill of 2048 tokens from producing
    /// 2048 × vocab floats.
    pub logits_indices: Vec<u32>,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
}

impl ForwardBatch {
    pub fn batch_size(&self) -> usize {
        self.cu_seqlens_q.len().saturating_sub(1)
    }

    pub fn num_tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Check every structural invariant a paged attention kernel relies on.
    ///
    /// These are the mistakes that produce plausible-but-wrong text rather
    /// than a crash: two tokens writing to one cache slot, a block table that
    /// does not cover the key length, positions that skip. Running this in
    /// tests turns every scheduler test into a block-accounting test as well.
    pub fn validate(&self, num_blocks: usize, block_size: usize) -> Result<()> {
        let bail = |m: String| Err(vapi_core::Error::Engine(m));

        if self.tokens.len() != self.positions.len() {
            return bail(format!(
                "tokens ({}) and positions ({}) disagree",
                self.tokens.len(),
                self.positions.len()
            ));
        }
        if self.tokens.len() != self.slot_mapping.len() {
            return bail(format!(
                "tokens ({}) and slot_mapping ({}) disagree",
                self.tokens.len(),
                self.slot_mapping.len()
            ));
        }
        if self.cu_seqlens_q.len() != self.cu_seqlens_k.len() {
            return bail("cu_seqlens_q and cu_seqlens_k differ in length".into());
        }
        if self.cu_seqlens_q.first() != Some(&0) && !self.cu_seqlens_q.is_empty() {
            return bail("cumulative lengths must start at 0".into());
        }
        if self.block_tables.len() != self.batch_size() {
            return bail(format!(
                "{} block tables for {} sequences",
                self.block_tables.len(),
                self.batch_size()
            ));
        }

        for w in self.cu_seqlens_q.windows(2) {
            if w[1] < w[0] {
                return bail("cu_seqlens_q is not monotonic".into());
            }
        }
        for w in self.cu_seqlens_k.windows(2) {
            if w[1] < w[0] {
                return bail("cu_seqlens_k is not monotonic".into());
            }
        }
        if let Some(&last) = self.cu_seqlens_q.last()
            && last as usize != self.tokens.len()
        {
            return bail(format!(
                "cu_seqlens_q ends at {last} but there are {} tokens",
                self.tokens.len()
            ));
        }

        for i in 0..self.batch_size() {
            let q = (self.cu_seqlens_q[i + 1] - self.cu_seqlens_q[i]) as usize;
            let k = (self.cu_seqlens_k[i + 1] - self.cu_seqlens_k[i]) as usize;
            if q > k {
                return bail(format!(
                    "sequence {i} has {q} query tokens but only {k} key tokens; \
                     a token cannot attend to fewer positions than it contributes"
                ));
            }
            // The block table must cover every key position, cached prefix
            // included — this is the check that catches a prefix-cache hit
            // whose blocks were not carried into the batch.
            let need = k.div_ceil(block_size);
            if self.block_tables[i].len() < need {
                return bail(format!(
                    "sequence {i} needs {need} blocks to cover {k} keys but its table has {}",
                    self.block_tables[i].len()
                ));
            }
            for &b in &self.block_tables[i] {
                if b as usize >= num_blocks {
                    return bail(format!("sequence {i} references block {b} of {num_blocks}"));
                }
            }
            // Positions must be contiguous within a sequence; a gap means the
            // sequence's RoPE offsets and its cache slots have diverged.
            let lo = self.cu_seqlens_q[i] as usize;
            for j in lo + 1..lo + q {
                if self.positions[j] != self.positions[j - 1] + 1 {
                    return bail(format!(
                        "sequence {i} has non-contiguous positions at {}: {} then {}",
                        j,
                        self.positions[j - 1],
                        self.positions[j]
                    ));
                }
            }
        }

        // No two tokens may target the same cache slot; one would overwrite
        // the other's K/V and the loser would attend to the wrong state.
        let mut slots = self.slot_mapping.clone();
        slots.sort_unstable();
        if let Some(w) = slots.windows(2).find(|w| w[0] == w[1]) {
            return bail(format!("two tokens both write KV cache slot {}", w[0]));
        }
        if let Some(&s) = slots.last()
            && s as usize >= num_blocks * block_size
        {
            return bail(format!("slot {s} is past the end of the cache"));
        }

        for &i in &self.logits_indices {
            if i as usize >= self.tokens.len() {
                return bail(format!("logits index {i} is past the end of the batch"));
            }
        }
        Ok(())
    }
}

/// Logits for the sampled positions of one step: `[num_logits_indices, vocab]`
/// in row-major order.
#[derive(Clone, Debug)]
pub struct Logits {
    pub data: Vec<f32>,
    pub vocab_size: usize,
}

impl Logits {
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.vocab_size..(i + 1) * self.vocab_size]
    }

    /// The row as a mutable slice, so a sampler can apply penalties and
    /// temperature in place instead of copying 512 KB per row first.
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        &mut self.data[i * self.vocab_size..(i + 1) * self.vocab_size]
    }

    pub fn rows(&self) -> usize {
        self.data.len().checked_div(self.vocab_size).unwrap_or(0)
    }
}

/// One sampled row's candidate tokens, selected on the device: every token
/// whose logit is at or above a per-row threshold, as `(token_id, raw
/// logit)` in no particular order, plus the row's full softmax denominator
/// so the host can normalise exactly without the tail.
///
/// `sum` is `Σ_v exp((logit_v - max) / temperature)` over the *whole*
/// vocabulary, with `max` the row maximum (`self.max`).
#[derive(Clone, Debug, Default)]
pub struct RowCandidates {
    pub entries: Vec<(u32, f32)>,
    pub max: f32,
    pub sum: f32,
}

/// One row's logits as the sampler receives them: the whole vocabulary,
/// or only the candidates when the backend could select them exactly.
#[derive(Clone, Debug)]
pub enum RowLogits {
    Full(Vec<f32>),
    Candidates(RowCandidates),
    /// Drawn on the device, see [`GumbelDraw`].
    Token(u32),
}

/// What a step produced for the sampler: every logit for every row, or a
/// per-row mix of full rows and device-selected candidates.
#[derive(Clone, Debug)]
pub enum StepLogits {
    Full(Logits),
    Rows(Vec<RowLogits>),
}

/// One row's logits as the sampler receives them, or the token itself when
/// the backend drew it.
///
/// A device draw is a Gumbel-max sample: `argmax_v(logit_v / T + g_v)` with
/// `g_v` standard Gumbel noise from a counter-based hash of `(seed, draw,
/// v)`, which samples exactly from the tempered softmax and reproduces
/// for the same seed. It is asked for only when nothing but temperature
/// applies (no top-k, top-p, min-p), see [`GumbelDraw`].
#[derive(Clone, Debug, PartialEq)]
pub struct GumbelDraw {
    /// Per-sequence seed for the noise.
    pub seed: u64,
    /// Index of this draw within the sequence; the noise changes with it.
    pub draw: u64,
    /// `(token, occurrence count)` pairs the penalties apply to.
    pub observed: Vec<(u32, u32)>,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

/// Per-row requirement for [`ExecutionBackend::forward_candidates`].
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateNeed {
    /// `1 / temperature` used to judge the candidate window; 1.0 for greedy.
    pub inv_temperature: f32,
    /// The row is exactly served by its top `needed` tokens (top-k plus the
    /// tokens penalties can move), or, when 0, only by every token within
    /// the sampler's candidate window.
    pub needed: usize,
    /// When set, the backend may draw the token itself instead and return
    /// [`RowLogits::Token`].
    pub gumbel: Option<GumbelDraw>,
    /// The host needs the whole row (it will report logprobs), so no
    /// device-side shortcut may be taken for it.
    pub full: bool,
}

/// Model execution. The only trait a new device or model runtime has to
/// implement.
pub trait ExecutionBackend: Send {
    fn spec(&self) -> &ModelSpec;

    /// Run one batch and return logits for `batch.logits_indices` only.
    ///
    /// Implementations write each token's K/V into the paged cache at
    /// `batch.slot_mapping` as a side effect; the engine relies on that having
    /// happened before the next step.
    fn forward(&mut self, batch: &ForwardBatch) -> Result<Logits>;

    /// Run one batch and return the argmax token for each of
    /// `batch.logits_indices`, sampled on the device, without bringing the
    /// logits to the host. `Ok(None)` means the backend does not implement
    /// it and the engine should call [`forward`](Self::forward) instead.
    ///
    /// Only valid when every sampled row is greedy and penalty-free: the
    /// engine checks that before calling. For a 128K vocabulary this saves
    /// 512 KB of copy and a 128K-element scan per row per step.
    fn forward_greedy(&mut self, _batch: &ForwardBatch) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }

    /// Run one batch and return, per sampled row, only the tokens the
    /// sampler can need, selected on the device (see [`RowCandidates`]);
    /// or the full logits when the backend cannot guarantee that for every
    /// row. The default is the full logits. `needs` has one entry per
    /// `batch.logits_indices`.
    fn forward_candidates(
        &mut self,
        batch: &ForwardBatch,
        _needs: &[CandidateNeed],
    ) -> Result<StepLogits> {
        Ok(StepLogits::Full(self.forward(batch)?))
    }

    /// Number of KV blocks the backend has allocated.
    fn num_blocks(&self) -> usize;

    /// Bytes one KV block occupies across every layer, or 0 when this
    /// backend cannot move blocks in and out. The spill tier is off unless
    /// this is non-zero.
    fn block_bytes(&self) -> usize {
        0
    }

    /// Copy one block's KV out of the cache, for the spill tier. The layout
    /// is the backend's own; only the same backend reads it back.
    fn export_block(&self, _id: BlockId) -> Result<Vec<u8>> {
        Err(vapi_core::Error::Engine(
            "this backend cannot export blocks".into(),
        ))
    }

    /// Put previously exported bytes back into a block the caller owns.
    fn import_block(&mut self, _id: BlockId, _bytes: &[u8]) -> Result<()> {
        Err(vapi_core::Error::Engine(
            "this backend cannot import blocks".into(),
        ))
    }

    /// Copy blocks for copy-on-write, used when a shared partially-filled
    /// block is about to be written by one of its sharers.
    fn copy_blocks(&mut self, _pairs: &[(BlockId, BlockId)]) -> Result<()> {
        Ok(())
    }
}

/// A backend with no model behind it.
///
/// This is the single highest-leverage piece of test infrastructure in the
/// project: it makes the scheduler, the cache, the NATS protocol and the HTTP
/// surface all testable in milliseconds on a machine with no GPU. It also
/// *validates every batch it is given*, so a scheduler test that never looks
/// at a tensor still catches a malformed block table or a duplicated cache
/// slot.
pub struct MockBackend {
    spec: ModelSpec,
    num_blocks: usize,
    block_size: usize,
    /// Emit this token id at each step, cycling. Lets a test script an exact
    /// output including when EOS lands.
    script: Vec<u32>,
    step: usize,
    /// Artificial per-step latency. Lets a laptop simulate a GPU's step time,
    /// so scheduler behaviour, cancellation and backpressure can be exercised
    /// at realistic pacing without a GPU.
    step_delay: std::time::Duration,
    /// Every batch this backend has been handed, kept only when
    /// [`MockBackend::recording`] was called. A long-running mock worker
    /// would otherwise grow this without bound.
    pub batches_seen: Vec<ForwardBatch>,
    record: bool,
    /// Fail the forward on this step index (0-based), for the failure
    /// policy tests.
    fail_on_step: Option<usize>,
    /// Stand-in for device KV: whatever was last written to each block.
    /// Lets the spill tier be exercised without a GPU.
    block_store: HashMap<u32, Vec<u8>>,
    /// Non-zero once [`MockBackend::with_movable_blocks`] is called.
    block_bytes: usize,
}

impl MockBackend {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        Self {
            spec: ModelSpec::tiny(),
            num_blocks,
            block_size,
            script: Vec::new(),
            step: 0,
            step_delay: std::time::Duration::ZERO,
            batches_seen: Vec::new(),
            record: false,
            fail_on_step: None,
            block_store: HashMap::new(),
            block_bytes: 0,
        }
    }

    /// Keep a copy of every batch in `batches_seen`, for tests that inspect
    /// how the scheduler shaped its work.
    pub fn recording(mut self) -> Self {
        self.record = true;
        self
    }

    /// Fix the exact sequence of tokens this backend will produce.
    pub fn with_script(mut self, script: impl IntoIterator<Item = u32>) -> Self {
        self.script = script.into_iter().collect();
        self
    }

    pub fn with_spec(mut self, spec: ModelSpec) -> Self {
        self.spec = spec;
        self
    }

    /// Simulate a slower device.
    pub fn with_step_delay(mut self, delay: std::time::Duration) -> Self {
        self.step_delay = delay;
        self
    }

    /// Make the forward of step `step` (0-based) return an error.
    pub fn failing_at(mut self, step: usize) -> Self {
        self.fail_on_step = Some(step);
        self
    }

    /// Pretend blocks of `bytes` bytes can be moved in and out, so the spill
    /// tier can be tested without a device. Each block's "KV" is whatever
    /// the forward last stamped into it.
    pub fn with_movable_blocks(mut self, bytes: usize) -> Self {
        self.block_bytes = bytes;
        self
    }

    /// What the forward writes into a block: the step that wrote it, so a
    /// test can tell recomputed content from restored content.
    fn stamp(&self, block: u32) -> Vec<u8> {
        let mut v = vec![0u8; self.block_bytes];
        for (i, b) in v.iter_mut().enumerate() {
            *b = (block as usize + i + self.step) as u8;
        }
        v
    }

    fn next_token(&mut self) -> u32 {
        if self.script.is_empty() {
            // Deterministic but content-dependent, so a test can still assert
            // that two identical prompts produce identical output.
            return (self.step as u32 * 7 + 1) % self.spec.vocab_size as u32;
        }
        self.script[self.step % self.script.len()]
    }
}

impl ExecutionBackend for MockBackend {
    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    fn export_block(&self, id: BlockId) -> Result<Vec<u8>> {
        if self.block_bytes == 0 {
            return Err(vapi_core::Error::Engine("blocks are not movable".into()));
        }
        Ok(self
            .block_store
            .get(&id.0)
            .cloned()
            .unwrap_or_else(|| vec![0u8; self.block_bytes]))
    }

    fn import_block(&mut self, id: BlockId, bytes: &[u8]) -> Result<()> {
        if self.block_bytes == 0 {
            return Err(vapi_core::Error::Engine("blocks are not movable".into()));
        }
        if bytes.len() != self.block_bytes {
            return Err(vapi_core::Error::Engine(format!(
                "block is {} bytes, got {}",
                self.block_bytes,
                bytes.len()
            )));
        }
        self.block_store.insert(id.0, bytes.to_vec());
        Ok(())
    }

    fn forward(&mut self, batch: &ForwardBatch) -> Result<Logits> {
        batch.validate(self.num_blocks, self.block_size)?;
        if self.fail_on_step == Some(self.step) {
            self.step += 1;
            return Err(vapi_core::Error::Engine("injected step failure".into()));
        }
        if !self.step_delay.is_zero() {
            std::thread::sleep(self.step_delay);
        }
        if self.record {
            self.batches_seen.push(batch.clone());
        }

        if self.block_bytes > 0 {
            // Stand in for writing K/V: stamp every block this batch touches.
            for slot in &batch.slot_mapping {
                let block = slot / self.block_size as u32;
                let stamp = self.stamp(block);
                self.block_store.insert(block, stamp);
            }
        }

        let vocab = self.spec.vocab_size;
        let rows = batch.logits_indices.len();
        let mut data = vec![0.0f32; rows * vocab];
        for r in 0..rows {
            let t = self.next_token() as usize % vocab;
            // One-hot: greedy and sampling both land on the scripted token, so
            // a test can pin output without disabling the sampler.
            data[r * vocab + t] = 100.0;
        }
        self.step += 1;
        Ok(Logits {
            data,
            vocab_size: vocab,
        })
    }
}
