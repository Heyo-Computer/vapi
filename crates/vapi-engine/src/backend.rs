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

    pub fn rows(&self) -> usize {
        self.data.len().checked_div(self.vocab_size).unwrap_or(0)
    }
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

    /// Number of KV blocks the backend has allocated.
    fn num_blocks(&self) -> usize;

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
    pub batches_seen: Vec<ForwardBatch>,
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
        }
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

    fn forward(&mut self, batch: &ForwardBatch) -> Result<Logits> {
        batch.validate(self.num_blocks, self.block_size)?;
        if !self.step_delay.is_zero() {
            std::thread::sleep(self.step_delay);
        }
        self.batches_seen.push(batch.clone());

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
