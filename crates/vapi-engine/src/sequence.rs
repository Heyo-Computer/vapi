use vapi_cache::BlockId;
use vapi_core::{FinishReason, RequestId, SamplingParams};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeqStatus {
    /// Admitted but not yet prefilled.
    Waiting,
    /// Prefill in progress; `num_computed` has not yet reached the prompt end.
    Prefilling,
    /// Generating, one token per step.
    Decoding,
    /// Preempted; blocks released, will be recomputed from scratch.
    Preempted,
    Finished(FinishReason),
}

impl SeqStatus {
    pub fn is_finished(&self) -> bool {
        matches!(self, Self::Finished(_))
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Prefilling | Self::Decoding)
    }
}

/// One in-flight generation.
///
/// `num_computed` is the load-bearing field: it is how many of this sequence's
/// tokens already have K/V in the cache, counting both a cache-hit prefix it
/// never computed and chunks it has prefilled so far. Every position, slot and
/// cumulative length in a batch derives from it, so an off-by-one here is the
/// single most likely cause of output that is fluent but wrong.
#[derive(Debug)]
pub struct Sequence {
    pub request_id: RequestId,
    pub params: SamplingParams,
    pub status: SeqStatus,

    /// Prompt followed by everything generated so far.
    tokens: Vec<u32>,
    prompt_len: usize,
    num_computed: usize,
    /// How many leading tokens are known text to prefill: the prompt, or
    /// after a preemption the prompt plus what had been generated. Without
    /// this a recomputed sequence replays its own output one token per step.
    prefill_target: usize,

    /// Cache blocks backing this sequence, in logical order.
    pub blocks: Vec<BlockId>,
    /// Prompt tokens served from the shared prefix cache, for reporting.
    pub cached_prefix_tokens: usize,
    /// How many of `blocks` came from the cache and are therefore already
    /// published — they must not be re-published on completion.
    pub num_cached_blocks: usize,

    pub namespace: String,
}

impl Sequence {
    pub fn new(
        request_id: RequestId,
        prompt: Vec<u32>,
        params: SamplingParams,
        namespace: String,
    ) -> Self {
        let prompt_len = prompt.len();
        Self {
            request_id,
            params,
            status: SeqStatus::Waiting,
            tokens: prompt,
            prompt_len,
            num_computed: 0,
            prefill_target: prompt_len,
            blocks: Vec::new(),
            cached_prefix_tokens: 0,
            num_cached_blocks: 0,
            namespace,
        }
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    pub fn total_len(&self) -> usize {
        self.tokens.len()
    }

    pub fn num_computed(&self) -> usize {
        self.num_computed
    }

    pub fn generated(&self) -> &[u32] {
        &self.tokens[self.prompt_len..]
    }

    pub fn num_generated(&self) -> usize {
        self.tokens.len() - self.prompt_len
    }

    /// Known tokens still needing a forward pass before decoding can start.
    pub fn remaining_prefill(&self) -> usize {
        self.prefill_target.saturating_sub(self.num_computed)
    }

    /// Length of the known-text prefix that prefill covers. Equal to the
    /// prompt length unless the sequence is being recomputed.
    pub fn prefill_target(&self) -> usize {
        self.prefill_target
    }

    pub fn needs_prefill(&self) -> bool {
        self.remaining_prefill() > 0
    }

    /// Adopt a cache-hit prefix: these tokens already have K/V, so they are
    /// computed without ever being forwarded.
    pub fn adopt_cached_prefix(&mut self, blocks: Vec<BlockId>, tokens: usize) {
        debug_assert_eq!(
            self.num_computed, 0,
            "prefix must be adopted before any compute"
        );
        debug_assert!(tokens <= self.prefill_target);
        self.num_cached_blocks = blocks.len();
        self.blocks = blocks;
        self.cached_prefix_tokens = tokens;
        self.num_computed = tokens;
    }

    /// Record that `n` more tokens were forwarded.
    pub fn advance_computed(&mut self, n: usize) {
        self.num_computed += n;
        debug_assert!(
            self.num_computed <= self.tokens.len(),
            "computed {} past sequence length {}",
            self.num_computed,
            self.tokens.len()
        );
    }

    /// Append a sampled token.
    pub fn push_token(&mut self, t: u32) {
        self.tokens.push(t);
    }

    /// Drop all cache state, so the sequence restarts from its prompt.
    /// Generated tokens are kept — they are part of the prompt on the retry.
    pub fn reset_for_recompute(&mut self) {
        self.blocks.clear();
        self.num_computed = 0;
        self.num_cached_blocks = 0;
        self.cached_prefix_tokens = 0;
        self.prefill_target = self.tokens.len();
        self.status = SeqStatus::Preempted;
    }

    /// Whether generation should stop, given the model's EOS ids.
    pub fn check_finished(&self, eos: &[u32], max_context: usize) -> Option<FinishReason> {
        if let Some(&last) = self.tokens.last()
            && self.num_generated() > 0
            && eos.contains(&last)
        {
            return Some(FinishReason::Stop);
        }
        if self
            .params
            .stop
            .stop_token_ids
            .contains(self.tokens.last()?)
            && self.num_generated() > 0
        {
            return Some(FinishReason::Stop);
        }
        if self.num_generated() >= self.params.max_tokens {
            return Some(FinishReason::Length);
        }
        if self.total_len() >= max_context {
            return Some(FinishReason::Length);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(prompt: Vec<u32>) -> Sequence {
        Sequence::new(
            RequestId::new(),
            prompt,
            SamplingParams::default(),
            "global".into(),
        )
    }

    #[test]
    fn a_fresh_sequence_must_prefill_its_whole_prompt() {
        let s = seq(vec![1, 2, 3, 4]);
        assert_eq!(s.remaining_prefill(), 4);
        assert!(s.needs_prefill());
        assert_eq!(s.num_generated(), 0);
    }

    #[test]
    fn a_cache_hit_removes_prefill_work() {
        let mut s = seq(vec![1, 2, 3, 4]);
        s.adopt_cached_prefix(vec![BlockId(0)], 2);
        assert_eq!(s.num_computed(), 2);
        assert_eq!(s.remaining_prefill(), 2, "only the uncached suffix is left");
        assert_eq!(s.cached_prefix_tokens, 2);
    }

    #[test]
    fn a_fully_cached_prompt_needs_no_forward_pass_for_its_prefix() {
        let mut s = seq(vec![1, 2, 3, 4]);
        s.adopt_cached_prefix(vec![BlockId(0), BlockId(1)], 4);
        assert_eq!(s.remaining_prefill(), 0);
        assert!(!s.needs_prefill());
    }

    #[test]
    fn generated_tokens_are_separable_from_the_prompt() {
        let mut s = seq(vec![1, 2]);
        s.advance_computed(2);
        s.push_token(9);
        s.push_token(8);
        assert_eq!(s.generated(), &[9, 8]);
        assert_eq!(s.prompt_len(), 2);
        assert_eq!(s.total_len(), 4);
    }

    #[test]
    fn eos_stops_generation_but_an_eos_in_the_prompt_does_not() {
        let mut s = seq(vec![1, 0, 2]); // 0 is EOS, but it is part of the prompt
        assert_eq!(s.check_finished(&[0], 100), None);
        s.push_token(0);
        assert_eq!(s.check_finished(&[0], 100), Some(FinishReason::Stop));
    }

    #[test]
    fn max_tokens_stops_generation() {
        let mut s = seq(vec![1]);
        s.params.max_tokens = 2;
        s.push_token(5);
        assert_eq!(s.check_finished(&[0], 100), None);
        s.push_token(6);
        assert_eq!(s.check_finished(&[0], 100), Some(FinishReason::Length));
    }

    #[test]
    fn running_out_of_context_stops_generation() {
        let mut s = seq(vec![1, 2, 3]);
        s.params.max_tokens = 1000;
        s.push_token(4);
        assert_eq!(s.check_finished(&[0], 4), Some(FinishReason::Length));
    }

    #[test]
    fn preemption_discards_cache_state_but_keeps_progress() {
        let mut s = seq(vec![1, 2]);
        s.adopt_cached_prefix(vec![BlockId(0)], 2);
        s.push_token(7);
        s.reset_for_recompute();

        assert_eq!(s.num_computed(), 0, "must recompute from scratch");
        assert!(s.blocks.is_empty());
        // The token it already produced is not thrown away; it becomes part of
        // what gets recomputed, so the client never sees a token retracted.
        assert_eq!(s.generated(), &[7]);
        assert_eq!(s.status, SeqStatus::Preempted);
        // The generated token is prefilled along with the prompt on the
        // retry, in chunks, rather than replayed one decode step at a time.
        assert_eq!(s.remaining_prefill(), 3);
        assert_eq!(s.prompt_len(), 2, "reporting still sees the real prompt");
    }
}
