//! The single-pass engine.
//!
//! What replaces continuous batching when nothing streams. A request arrives
//! as a set of independent sequences, one per question; a step takes as many
//! of them as the budget allows, runs one forward pass, and answers every
//! request whose rows are all scored.
//!
//! There is no collection window and no artificial delay. Batching still
//! emerges under load for the same reason it does on the decode path: while a
//! pass is running, arrivals queue up behind it, and the next pass takes them
//! all. An idle worker answers a single request immediately, which is the
//! behaviour that matters at the latency this model is chosen for.
//!
//! Rows are taken in arrival order rather than sorted by length. Sorting would
//! pack each pass more tightly — a pass costs `rows × longest row` — but it
//! starves long questions behind an unbounded supply of short ones, and the
//! rows of one request are near-identical in length anyway, since they share a
//! state.

use std::collections::HashMap;
use std::collections::VecDeque;

use vapi_core::{Error, RequestId, Result};
use vapi_engine::{EncoderBackend, EncoderBatch, EncoderRow};
use vapi_proto::{Delta, DeltaSeq, Job, JobKind, RowScores};

use crate::engine::StepOutput;

/// A request waiting for, or part-way through, its answers.
struct Pending {
    request_id: RequestId,
    rows: Vec<EncoderRow>,
    /// One slot per row, filled as passes complete.
    scores: Vec<Option<RowScores>>,
    prompt_tokens: usize,
}

impl Pending {
    fn unanswered(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.rows.len()).filter(|&i| self.scores[i].is_none())
    }

    fn complete(&self) -> bool {
        self.scores.iter().all(Option::is_some)
    }
}

pub struct DecisionEngine {
    backend: Box<dyn EncoderBackend>,
    queue: VecDeque<Pending>,
    delta_seq: HashMap<RequestId, DeltaSeq>,
    /// Requests admitted at once.
    max_requests: usize,
    /// Rows in one pass.
    max_rows: usize,
    /// Padded positions in one pass: `rows × longest row`. The real cost of a
    /// pass, and the reason a long row does not silently multiply the work of
    /// every short row sharing its batch.
    max_batched_tokens: usize,
    worker_id: String,
}

impl DecisionEngine {
    // Unused when the binary is built without `candle`: nothing can load a
    // decision checkpoint, so nothing constructs one outside the tests.
    #[cfg_attr(not(feature = "candle"), allow(dead_code))]
    pub fn new(
        backend: Box<dyn EncoderBackend>,
        max_requests: usize,
        max_rows: usize,
        max_batched_tokens: usize,
        worker_id: String,
    ) -> Self {
        Self {
            backend,
            queue: VecDeque::new(),
            delta_seq: HashMap::new(),
            max_requests: max_requests.max(1),
            max_rows: max_rows.max(1),
            max_batched_tokens: max_batched_tokens.max(1),
            worker_id,
        }
    }

    pub fn admit(&mut self, job: Job) -> Result<()> {
        if job.kind != JobKind::Decision {
            return Err(Error::InvalidRequest(format!(
                "this worker serves decisions; it was handed a {:?} job",
                job.kind
            )));
        }
        let parts = job
            .decision_rows()
            .map_err(|e| Error::InvalidRequest(format!("malformed decision job: {e}")))?;
        if parts.is_empty() {
            return Err(Error::InvalidRequest(
                "decision job has no questions".into(),
            ));
        }
        let max_context = self.backend.spec().max_context;
        let rows: Vec<EncoderRow> = parts
            .iter()
            .map(|(tokens, row)| EncoderRow {
                tokens: tokens.to_vec(),
                markers: row.markers.clone(),
                qtype: row.qtype,
            })
            .collect();
        // Validate before anything is queued, so a malformed request fails
        // its own admission rather than the step of whoever it batched with.
        EncoderBatch::new(rows.clone()).validate(max_context)?;

        let prompt_tokens = job.prompt_tokens.len();
        self.queue.push_back(Pending {
            request_id: job.request_id,
            scores: vec![None; rows.len()],
            rows,
            prompt_tokens,
        });
        Ok(())
    }

    pub fn cancel(&mut self, request_id: &RequestId) -> bool {
        let before = self.queue.len();
        self.queue.retain(|p| &p.request_id != request_id);
        before != self.queue.len()
    }

    pub fn fail_all(&mut self, message: &str) -> StepOutput {
        let mut out = StepOutput {
            events: Vec::new(),
            completed: Vec::new(),
            did_work: false,
        };
        for pending in self.queue.drain(..) {
            out.events.push((
                pending.request_id.clone(),
                Delta::Failed {
                    message: message.to_string(),
                },
            ));
            out.completed.push(pending.request_id);
        }
        out.did_work = !out.events.is_empty();
        out
    }

    pub fn is_idle(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn headroom(&self) -> usize {
        self.max_requests.saturating_sub(self.queue.len())
    }

    /// `(running, waiting, kv utilization, prefix hit rate)`.
    ///
    /// The last two are zero and always will be: an encoder has no KV cache,
    /// and a prefix cache would be *wrong* here rather than merely absent —
    /// attention is bidirectional, so a prefix's representation depends on the
    /// state that follows it.
    pub fn stats(&self) -> (usize, usize, f32, f32) {
        let started = self
            .queue
            .iter()
            .filter(|p| p.scores.iter().any(Option::is_some))
            .count();
        (started, self.queue.len() - started, 0.0, 0.0)
    }

    pub fn step(&mut self) -> Result<StepOutput> {
        let mut out = StepOutput {
            events: Vec::new(),
            completed: Vec::new(),
            did_work: false,
        };
        let picked = self.select();
        if picked.is_empty() {
            return Ok(out);
        }

        // `Started` is what tells the gateway the difference between "still
        // queued" and "running but slow", and it is the same contract the
        // decode path publishes.
        for &(q, r) in &picked {
            if self.queue[q].scores.iter().all(Option::is_none) && r == 0 {
                out.events.push((
                    self.queue[q].request_id.clone(),
                    Delta::Started {
                        worker_id: self.worker_id.clone(),
                        cached_prefix_tokens: 0,
                    },
                ));
            }
        }

        let batch = EncoderBatch::new(
            picked
                .iter()
                .map(|&(q, r)| self.queue[q].rows[r].clone())
                .collect(),
        );
        let rows = batch.num_rows();
        let width = batch.max_len();
        let started = std::time::Instant::now();
        let answers = self.backend.forward(&batch)?;
        if answers.rows.len() != rows {
            return Err(Error::Engine(format!(
                "backend answered {} of {rows} rows",
                answers.rows.len()
            )));
        }
        metrics::histogram!("vapi_decision_pass_seconds").record(started.elapsed().as_secs_f64());
        metrics::counter!("vapi_decision_rows_total").increment(rows as u64);
        metrics::counter!("vapi_decision_padded_tokens_total").increment((rows * width) as u64);
        tracing::debug!(
            rows,
            width,
            ms = started.elapsed().as_millis() as u64,
            "decision pass"
        );

        for (&(q, r), scores) in picked.iter().zip(answers.rows) {
            self.queue[q].scores[r] = Some(RowScores {
                logits: scores.logits,
                act: scores.act,
            });
        }

        // Answer every request whose rows are all in, keeping arrival order.
        let mut i = 0;
        while i < self.queue.len() {
            if !self.queue[i].complete() {
                i += 1;
                continue;
            }
            let pending = self.queue.remove(i).expect("index checked");
            out.events.push((
                pending.request_id.clone(),
                Delta::Decided {
                    rows: pending.scores.into_iter().flatten().collect(),
                    prompt_tokens: pending.prompt_tokens,
                },
            ));
            out.completed.push(pending.request_id);
        }
        out.did_work = true;
        Ok(out)
    }

    /// Which `(request, row)` pairs the next pass will carry.
    fn select(&self) -> Vec<(usize, usize)> {
        let mut picked = Vec::new();
        let mut width = 0usize;
        for (q, pending) in self.queue.iter().enumerate() {
            for r in pending.unanswered() {
                let len = pending.rows[r].len();
                let next_width = width.max(len);
                let cost = (picked.len() + 1) * next_width;
                if !picked.is_empty()
                    && (picked.len() >= self.max_rows || cost > self.max_batched_tokens)
                {
                    return picked;
                }
                width = next_width;
                picked.push((q, r));
            }
        }
        picked
    }

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
    use vapi_core::{ModelId, QuestionType, SamplingParams};
    use vapi_engine::{EncoderSpec, MockEncoder};
    use vapi_proto::DecisionRow;

    fn engine(max_rows: usize, max_tokens: usize) -> DecisionEngine {
        DecisionEngine::new(
            Box::new(MockEncoder::new().with_spec(EncoderSpec {
                max_context: 512,
                ..EncoderSpec::tiny()
            })),
            8,
            max_rows,
            max_tokens,
            "w-test".into(),
        )
    }

    /// A job of `lens.len()` questions, each with two options.
    fn job(id: &str, lens: &[usize]) -> Job {
        let mut tokens = Vec::new();
        let mut rows = Vec::new();
        for (i, &len) in lens.iter().enumerate() {
            tokens.extend((0..len).map(|t| (t + i * 7) as u32));
            rows.push(DecisionRow {
                len: len as u32,
                markers: vec![1, 2],
                qtype: QuestionType::Choice,
            });
        }
        Job {
            request_id: RequestId::parse(id).unwrap(),
            model: ModelId("m".into()),
            kind: JobKind::Decision,
            prompt_tokens: tokens,
            params: SamplingParams::default(),
            namespace: "global".into(),
            reply_to: format!("vapi.stream.{id}"),
            enqueued_at_ms: 0,
            rows,
        }
    }

    fn decided(out: &StepOutput, id: &str) -> Option<Vec<RowScores>> {
        out.events.iter().find_map(|(rid, d)| match d {
            Delta::Decided { rows, .. } if rid.as_str() == id => Some(rows.clone()),
            _ => None,
        })
    }

    #[test]
    fn every_question_is_answered_in_one_pass() {
        let mut e = engine(32, 4096);
        e.admit(job("r1", &[12, 12, 12, 12])).unwrap();
        let out = e.step().unwrap();
        let rows = decided(&out, "r1").expect("answered");
        assert_eq!(rows.len(), 4);
        assert_eq!(out.completed.len(), 1);
        assert!(e.is_idle(), "nothing should be left behind");
    }

    #[test]
    fn a_request_is_answered_only_when_all_of_its_rows_are() {
        // Two passes, one answer: a partial reply would be a request whose
        // caller sees some of its questions.
        let mut e = engine(2, 4096);
        e.admit(job("r1", &[10, 10, 10])).unwrap();
        let first = e.step().unwrap();
        assert!(decided(&first, "r1").is_none(), "answered too early");
        let second = e.step().unwrap();
        assert_eq!(decided(&second, "r1").unwrap().len(), 3);
    }

    #[test]
    fn waiting_requests_join_the_next_pass() {
        // This is the whole batching story: no window, no delay, but under
        // load a pass carries everything that queued behind the last one.
        let mut e = engine(32, 4096);
        e.admit(job("r1", &[10, 10])).unwrap();
        e.admit(job("r2", &[10])).unwrap();
        e.admit(job("r3", &[10])).unwrap();
        let out = e.step().unwrap();
        assert_eq!(out.completed.len(), 3, "all three in one pass");
    }

    #[test]
    fn a_long_row_does_not_drag_a_wide_batch_along_with_it() {
        // Cost is rows × longest, so the budget has to be checked against the
        // width the new row would create, not the width it had before.
        let mut e = engine(32, 256);
        e.admit(job("r1", &[8, 8, 8, 8])).unwrap();
        e.admit(job("r2", &[200])).unwrap();
        let picked = e.select();
        assert_eq!(picked.len(), 4, "the long row starts its own pass");
    }

    #[test]
    fn the_first_pass_announces_the_request_started() {
        let mut e = engine(32, 4096);
        e.admit(job("r1", &[10])).unwrap();
        let out = e.step().unwrap();
        assert!(
            out.events
                .iter()
                .any(|(_, d)| matches!(d, Delta::Started { .. })),
            "a gateway timeout cannot tell queued from slow without this"
        );
    }

    #[test]
    fn a_generation_job_is_refused_rather_than_answered() {
        let mut e = engine(32, 4096);
        let mut j = job("r1", &[10]);
        j.kind = JobKind::Chat;
        let err = e.admit(j).unwrap_err().to_string();
        assert!(err.contains("serves decisions"), "{err}");
    }

    #[test]
    fn a_job_whose_rows_do_not_add_up_is_refused_at_admission() {
        let mut e = engine(32, 4096);
        let mut j = job("r1", &[10, 10]);
        j.rows[1].len = 40;
        let err = e.admit(j).unwrap_err().to_string();
        assert!(err.contains("malformed decision job"), "{err}");
        assert!(e.is_idle());
    }

    #[test]
    fn an_over_long_question_is_refused_at_admission() {
        // Not at step time, where it would fail every request batched with it.
        let mut e = engine(32, 4096);
        let err = e.admit(job("r1", &[600])).unwrap_err().to_string();
        assert!(err.contains("over the 512"), "{err}");
        assert!(e.is_idle());
    }

    #[test]
    fn a_cancelled_request_leaves_the_queue() {
        let mut e = engine(32, 4096);
        e.admit(job("r1", &[10])).unwrap();
        e.admit(job("r2", &[10])).unwrap();
        assert!(e.cancel(&RequestId::parse("r1").unwrap()));
        let out = e.step().unwrap();
        assert!(decided(&out, "r1").is_none());
        assert!(decided(&out, "r2").is_some());
    }

    #[test]
    fn a_failed_pass_fails_the_requests_in_it() {
        let mut e = DecisionEngine::new(
            Box::new(MockEncoder::new().failing_at(0)),
            8,
            32,
            4096,
            "w-test".into(),
        );
        e.admit(job("r1", &[10])).unwrap();
        assert!(e.step().is_err());
        let out = e.fail_all("device wedged");
        assert!(matches!(out.events[0].1, Delta::Failed { .. }));
        assert!(e.is_idle());
    }

    #[test]
    fn load_is_reported_without_pretending_there_is_a_kv_cache() {
        let mut e = engine(1, 4096);
        e.admit(job("r1", &[10, 10])).unwrap();
        e.admit(job("r2", &[10])).unwrap();
        let (running, waiting, kv, prefix) = e.stats();
        assert_eq!((running, waiting), (0, 2));
        e.step().unwrap();
        let (running, waiting, _, _) = e.stats();
        assert_eq!((running, waiting), (1, 1), "r1 is part-way through");
        assert_eq!((kv, prefix), (0.0, 0.0));
    }
}
