//! The single-pass execution path.
//!
//! A decision or classification model is a bidirectional encoder: it reads a
//! whole sequence at once and answers from it, with no KV cache, no sampling
//! and no second step. That makes almost all of [`crate::backend`] wrong for
//! it — six of [`ForwardBatch`](crate::ForwardBatch)'s nine fields describe a
//! paged cache that does not exist here — so this is a separate trait rather
//! than a widening of that one. Nothing on the decode path should grow an
//! `Option` because encoders exist.
//!
//! Two consequences of bidirectionality are worth stating where they will be
//! read. Prefix caching is not merely useless but **wrong**: a token's
//! representation depends on the tokens after it, so a prefix computed under
//! one suffix is not the same tensor under another. And there is no
//! continuous batching to do, because nothing joins a batch part-way through;
//! what replaces it is a short collection window and length bucketing.

use vapi_core::{QuestionType, Result};

/// Static description of a loaded encoder.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderSpec {
    pub hidden_size: usize,
    /// Longest sequence the model can be given, in tokens.
    pub max_context: usize,
    /// Of that, how much the question and its option markers may spend. The
    /// remainder is the state. Both are the checkpoint's own numbers: the
    /// scorer was trained against sequences shaped this way.
    pub head_context: usize,
    pub pad_token_id: u32,
}

impl EncoderSpec {
    /// A small spec for tests.
    pub fn tiny() -> Self {
        Self {
            hidden_size: 16,
            max_context: 128,
            head_context: 48,
            pad_token_id: 0,
        }
    }
}

/// One question: the sequence it became, and where its option markers sit.
///
/// Marker positions are row-relative indices into `tokens`. They are carried
/// rather than re-derived because finding them again means scanning for a
/// mask token, and the sequence legitimately contains that token inside a
/// user's state.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderRow {
    pub tokens: Vec<u32>,
    pub markers: Vec<u32>,
    pub qtype: QuestionType,
}

impl EncoderRow {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn options(&self) -> usize {
        self.markers.len()
    }
}

/// A set of questions answered together in one forward pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EncoderBatch {
    pub rows: Vec<EncoderRow>,
}

impl EncoderBatch {
    pub fn new(rows: Vec<EncoderRow>) -> Self {
        Self { rows }
    }

    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    pub fn num_tokens(&self) -> usize {
        self.rows.iter().map(EncoderRow::len).sum()
    }

    /// Longest row. What a padded backend pads to, and what decides the cost
    /// of the pass: a batch is `num_rows * max_len` positions of work.
    pub fn max_len(&self) -> usize {
        self.rows.iter().map(EncoderRow::len).max().unwrap_or(0)
    }

    pub fn max_options(&self) -> usize {
        self.rows.iter().map(EncoderRow::options).max().unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Check what a backend would otherwise trust silently.
    ///
    /// A marker past the end of its row reads whatever padding holds, which
    /// is a confidently wrong answer rather than a crash — the same class of
    /// bug `ForwardBatch::validate` exists for.
    pub fn validate(&self, max_context: usize) -> Result<()> {
        let bail = |m: String| Err(vapi_core::Error::Engine(m));
        for (i, row) in self.rows.iter().enumerate() {
            if row.tokens.is_empty() {
                return bail(format!("row {i} has no tokens"));
            }
            if row.tokens.len() > max_context {
                return bail(format!(
                    "row {i} is {} tokens, over the {max_context} the encoder accepts",
                    row.tokens.len()
                ));
            }
            if row.markers.len() < 2 {
                return bail(format!(
                    "row {i} has {} option markers; every question has at least two",
                    row.markers.len()
                ));
            }
            let mut last: Option<u32> = None;
            for &m in &row.markers {
                if m as usize >= row.tokens.len() {
                    return bail(format!(
                        "row {i} marker at {m} is past its {} tokens",
                        row.tokens.len()
                    ));
                }
                if last.is_some_and(|l| m <= l) {
                    return bail(format!("row {i} markers are not strictly increasing"));
                }
                last = Some(m);
            }
        }
        Ok(())
    }
}

/// What one question's markers scored, before calibration.
///
/// Raw logits, not probabilities: the temperature that turns these into
/// honest probabilities is fitted per deployment and applied by the caller,
/// so the model must not bake in one it happens to ship with.
#[derive(Clone, Debug, PartialEq)]
pub struct RowScores {
    /// One logit per option marker, in the row's marker order.
    pub logits: Vec<f32>,
    /// The act head's probability of acting rather than escalating.
    pub act: f32,
}

/// One batch's answers, one entry per row, in the batch's order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MarkerLogits {
    pub rows: Vec<RowScores>,
}

/// A model that answers a whole request in one pass.
pub trait EncoderBackend: Send {
    fn spec(&self) -> &EncoderSpec;

    /// Run one batch. Returns one [`RowScores`] per row of `batch`, in order.
    fn forward(&mut self, batch: &EncoderBatch) -> Result<MarkerLogits>;
}

/// An encoder with no model behind it.
///
/// Plays the part [`MockBackend`](crate::MockBackend) plays for decode: it
/// makes the decision engine, the queue and the HTTP surface testable with
/// nothing downloaded, and it validates every batch it is handed.
///
/// Its scores are a deterministic function of the row's tokens and each
/// marker's position, so a test can assert an exact argmax without pinning a
/// magic number: [`MockEncoder::expected_argmax`] computes the same thing.
pub struct MockEncoder {
    spec: EncoderSpec,
    step_delay: std::time::Duration,
    fail_on_call: Option<usize>,
    calls: usize,
    /// Every batch seen, kept only after [`MockEncoder::recording`].
    pub batches_seen: Vec<EncoderBatch>,
    record: bool,
}

impl Default for MockEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl MockEncoder {
    pub fn new() -> Self {
        Self {
            spec: EncoderSpec::tiny(),
            step_delay: std::time::Duration::ZERO,
            fail_on_call: None,
            calls: 0,
            batches_seen: Vec::new(),
            record: false,
        }
    }

    pub fn with_spec(mut self, spec: EncoderSpec) -> Self {
        self.spec = spec;
        self
    }

    pub fn recording(mut self) -> Self {
        self.record = true;
        self
    }

    pub fn with_step_delay(mut self, delay: std::time::Duration) -> Self {
        self.step_delay = delay;
        self
    }

    /// Make call `call` (0-based) return an error.
    pub fn failing_at(mut self, call: usize) -> Self {
        self.fail_on_call = Some(call);
        self
    }

    pub fn calls(&self) -> usize {
        self.calls
    }

    fn score(row: &EncoderRow, option: usize) -> f32 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for t in &row.tokens {
            for b in t.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x1000_0000_01b3);
            }
        }
        h ^= (option as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        h = h.wrapping_mul(0x1000_0000_01b3);
        // A spread wide enough that a softmax over it is not uniform.
        ((h >> 40) as f32 / 16_777_216.0) * 8.0 - 4.0
    }

    /// Which option the mock will pick for this row.
    pub fn expected_argmax(row: &EncoderRow) -> usize {
        (0..row.options())
            .max_by(|&a, &b| {
                Self::score(row, a)
                    .partial_cmp(&Self::score(row, b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0)
    }
}

impl EncoderBackend for MockEncoder {
    fn spec(&self) -> &EncoderSpec {
        &self.spec
    }

    fn forward(&mut self, batch: &EncoderBatch) -> Result<MarkerLogits> {
        batch.validate(self.spec.max_context)?;
        let call = self.calls;
        self.calls += 1;
        if self.record {
            self.batches_seen.push(batch.clone());
        }
        if self.fail_on_call == Some(call) {
            return Err(vapi_core::Error::Engine(format!(
                "mock encoder failing on call {call}"
            )));
        }
        if !self.step_delay.is_zero() {
            std::thread::sleep(self.step_delay);
        }
        Ok(MarkerLogits {
            rows: batch
                .rows
                .iter()
                .map(|row| RowScores {
                    logits: (0..row.options()).map(|o| Self::score(row, o)).collect(),
                    act: 0.75,
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(tokens: &[u32], markers: &[u32]) -> EncoderRow {
        EncoderRow {
            tokens: tokens.to_vec(),
            markers: markers.to_vec(),
            qtype: QuestionType::Choice,
        }
    }

    #[test]
    fn the_mock_answers_one_row_per_question() {
        let batch = EncoderBatch::new(vec![
            row(&[1, 2, 3, 4, 5], &[1, 3]),
            row(&[9, 8, 7, 6, 5, 4], &[0, 2, 4]),
        ]);
        let out = MockEncoder::new().forward(&batch).unwrap();
        assert_eq!(out.rows.len(), 2);
        assert_eq!(out.rows[0].logits.len(), 2);
        assert_eq!(out.rows[1].logits.len(), 3);
    }

    #[test]
    fn the_mock_is_deterministic_and_its_pick_is_predictable() {
        let r = row(&[4, 5, 6, 7], &[0, 1, 2]);
        let batch = EncoderBatch::new(vec![r.clone()]);
        let a = MockEncoder::new().forward(&batch).unwrap();
        let b = MockEncoder::new().forward(&batch).unwrap();
        assert_eq!(a, b);
        let want = MockEncoder::expected_argmax(&r);
        let got = a.rows[0]
            .logits
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(got, want);
    }

    #[test]
    fn different_states_give_different_answers() {
        // A mock that answered the same thing regardless of input would let a
        // routing bug through every test that uses it.
        let picks: std::collections::HashSet<usize> = (0u32..24)
            .map(|i| MockEncoder::expected_argmax(&row(&[i, i + 1, i * 7 + 3], &[0, 1, 2])))
            .collect();
        assert!(picks.len() > 1, "the mock always picks {picks:?}");
    }

    #[test]
    fn a_marker_past_the_end_is_refused() {
        let batch = EncoderBatch::new(vec![row(&[1, 2, 3], &[1, 9])]);
        let err = MockEncoder::new().forward(&batch).unwrap_err().to_string();
        assert!(err.contains("past its 3 tokens"), "{err}");
    }

    #[test]
    fn markers_must_be_strictly_increasing() {
        // Out-of-order markers would still gather, silently pairing each
        // option with another option's hidden state.
        let batch = EncoderBatch::new(vec![row(&[1, 2, 3, 4], &[2, 1])]);
        let err = MockEncoder::new().forward(&batch).unwrap_err().to_string();
        assert!(err.contains("strictly increasing"), "{err}");
    }

    #[test]
    fn a_one_option_question_is_refused() {
        let batch = EncoderBatch::new(vec![row(&[1, 2, 3], &[1])]);
        let err = MockEncoder::new().forward(&batch).unwrap_err().to_string();
        assert!(err.contains("at least two"), "{err}");
    }

    #[test]
    fn an_over_long_row_is_refused() {
        let long: Vec<u32> = (0..200).collect();
        let batch = EncoderBatch::new(vec![row(&long, &[1, 2])]);
        let err = MockEncoder::new().forward(&batch).unwrap_err().to_string();
        assert!(err.contains("over the 128"), "{err}");
    }

    #[test]
    fn batch_shape_is_what_a_padded_backend_would_need() {
        let batch = EncoderBatch::new(vec![
            row(&[1, 2, 3], &[0, 1]),
            row(&[1, 2, 3, 4, 5, 6], &[0, 1, 2, 3]),
        ]);
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_tokens(), 9);
        assert_eq!(batch.max_len(), 6);
        assert_eq!(batch.max_options(), 4);
    }
}
