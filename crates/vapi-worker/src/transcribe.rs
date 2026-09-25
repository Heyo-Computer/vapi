//! The transcription engine.
//!
//! One clip per step, because a clip is not divisible the way a batch of
//! prompts is: the decode is lock-step with the audio, and a sixty-second
//! recording is four hundred and fifty sequential positions whose order is
//! the whole point. That makes a step long — fifteen seconds for a minute of
//! audio on this card — which is why the engine runs on its own thread and
//! the async side keeps acking progress while it works.
//!
//! What this does *not* do is batch two clips into one pass. That is the
//! whole of the throughput story and it needs the engine integration phase 9b
//! is for; until then a worker transcribes one request at a time, faster than
//! real time but one at a time.

use std::collections::{HashMap, VecDeque};

use vapi_core::{Error, RequestId, Result};
use vapi_proto::{Delta, DeltaSeq, Job, JobKind};

use crate::engine::StepOutput;

/// What a loaded speech model has to offer.
///
/// A trait so the engine is testable without eight gigabytes of weights, the
/// same reason the decode path has `MockBackend`.
pub trait SpeechBackend: Send {
    /// Transcribe 16 kHz mono samples into text.
    fn transcribe(&self, samples: &[f32]) -> Result<Transcript>;
    /// Sample rate the model's frontend expects.
    fn sample_rate(&self) -> usize;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    pub text: String,
    /// Decoder positions the audio occupied, one per 80 ms.
    pub positions: usize,
}

struct Pending {
    request_id: RequestId,
    samples: Vec<f32>,
    seconds: f32,
}

pub struct TranscriptionEngine {
    backend: Box<dyn SpeechBackend>,
    queue: VecDeque<Pending>,
    delta_seq: HashMap<RequestId, DeltaSeq>,
    max_queued: usize,
    worker_id: String,
}

impl TranscriptionEngine {
    // Unused when the binary is built without `candle`: nothing can load a
    // speech checkpoint, so nothing constructs one outside the tests.
    #[cfg_attr(not(feature = "candle"), allow(dead_code))]
    pub fn new(backend: Box<dyn SpeechBackend>, max_queued: usize, worker_id: String) -> Self {
        Self {
            backend,
            queue: VecDeque::new(),
            delta_seq: HashMap::new(),
            max_queued: max_queued.max(1),
            worker_id,
        }
    }

    pub fn admit(&mut self, job: Job) -> Result<()> {
        if job.kind != JobKind::Transcription {
            return Err(Error::InvalidRequest(format!(
                "this worker transcribes audio; it was handed a {:?} job",
                job.kind
            )));
        }
        let clip = job
            .audio
            .ok_or_else(|| Error::InvalidRequest("transcription job carries no audio".into()))?;
        let pcm = base64_decode(&clip.pcm)
            .ok_or_else(|| Error::InvalidRequest("audio is not valid base64".into()))?;
        let samples = vapi_audio::from_i16_le(&pcm);
        if samples.is_empty() {
            return Err(Error::InvalidRequest("audio is empty".into()));
        }
        // The gateway resamples, so a mismatch here means the two disagree
        // about the model — worth failing the request rather than quietly
        // transcribing at the wrong speed.
        let want = self.backend.sample_rate() as u32;
        if clip.sample_rate != want {
            return Err(Error::InvalidRequest(format!(
                "audio is {} Hz; this model wants {want} Hz",
                clip.sample_rate
            )));
        }
        let seconds = samples.len() as f32 / want as f32;
        self.queue.push_back(Pending {
            request_id: job.request_id,
            samples,
            seconds,
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
        self.max_queued.saturating_sub(self.queue.len())
    }

    /// `(running, waiting, kv utilization, prefix hit rate)`.
    ///
    /// The last two are zero and will stay zero: the decoder's cache is
    /// per-clip and lives for one step, and no two clips share a prefix.
    pub fn stats(&self) -> (usize, usize, f32, f32) {
        let running = usize::from(!self.queue.is_empty());
        (running, self.queue.len() - running, 0.0, 0.0)
    }

    pub fn step(&mut self) -> Result<StepOutput> {
        let mut out = StepOutput {
            events: Vec::new(),
            completed: Vec::new(),
            did_work: false,
        };
        let Some(pending) = self.queue.pop_front() else {
            return Ok(out);
        };
        out.events.push((
            pending.request_id.clone(),
            Delta::Started {
                worker_id: self.worker_id.clone(),
                cached_prefix_tokens: 0,
            },
        ));

        let started = std::time::Instant::now();
        let transcript = match self.backend.transcribe(&pending.samples) {
            Ok(t) => t,
            Err(e) => {
                // Put it back before returning. The runner's response to a
                // failed step is to fail everything in flight, and a request
                // already popped off the queue is not in flight as far as
                // that is concerned — the caller would wait for its timeout
                // and never learn why.
                self.queue.push_front(pending);
                return Err(e);
            }
        };
        let wall = started.elapsed().as_secs_f32();
        metrics::histogram!("vapi_transcription_seconds").record(wall as f64);
        metrics::counter!("vapi_transcribed_audio_seconds_total").increment(pending.seconds as u64);
        tracing::debug!(
            request_id = %pending.request_id,
            audio_seconds = pending.seconds,
            wall,
            realtime = pending.seconds / wall.max(1e-6),
            positions = transcript.positions,
            "transcribed"
        );

        out.events.push((
            pending.request_id.clone(),
            Delta::Transcribed {
                text: transcript.text,
                positions: transcript.positions,
                audio_seconds: pending.seconds,
            },
        ));
        out.completed.push(pending.request_id);
        out.did_work = true;
        Ok(out)
    }

    pub fn sequence(&mut self, rid: &RequestId, delta: Delta) -> vapi_proto::DeltaMsg {
        self.delta_seq.entry(rid.clone()).or_default().next(delta)
    }

    pub fn forget(&mut self, rid: &RequestId) {
        self.delta_seq.remove(rid);
    }
}

/// Standard base64, decoded without pulling the crate into this one.
fn base64_decode(text: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.decode(text).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vapi_core::{ModelId, SamplingParams};
    use vapi_proto::AudioClip;

    /// A backend with no model behind it: it reports how much audio it saw,
    /// which is enough to check the plumbing end to end.
    struct MockSpeech {
        rate: usize,
        fail: bool,
    }

    impl SpeechBackend for MockSpeech {
        fn transcribe(&self, samples: &[f32]) -> Result<Transcript> {
            if self.fail {
                return Err(Error::Engine("device wedged".into()));
            }
            Ok(Transcript {
                text: format!("heard {} samples", samples.len()),
                positions: samples.len() / 1280,
            })
        }
        fn sample_rate(&self) -> usize {
            self.rate
        }
    }

    fn engine(fail: bool) -> TranscriptionEngine {
        TranscriptionEngine::new(
            Box::new(MockSpeech { rate: 16_000, fail }),
            4,
            "w-test".into(),
        )
    }

    fn job(id: &str, samples: usize, rate: u32) -> Job {
        use base64::Engine;
        let pcm = vapi_audio::to_i16_le(&vec![0.25f32; samples]);
        Job {
            request_id: RequestId::parse(id).unwrap(),
            model: ModelId("m".into()),
            kind: JobKind::Transcription,
            prompt_tokens: Vec::new(),
            params: SamplingParams::default(),
            namespace: "global".into(),
            reply_to: format!("vapi.stream.{id}"),
            enqueued_at_ms: 0,
            rows: Vec::new(),
            audio: Some(AudioClip {
                sample_rate: rate,
                pcm: base64::engine::general_purpose::STANDARD.encode(&pcm),
            }),
        }
    }

    fn transcript(out: &StepOutput) -> Option<String> {
        out.events.iter().find_map(|(_, d)| match d {
            Delta::Transcribed { text, .. } => Some(text.clone()),
            _ => None,
        })
    }

    #[test]
    fn a_clip_is_transcribed_and_answered_once() {
        let mut e = engine(false);
        e.admit(job("r1", 16_000, 16_000)).unwrap();
        let out = e.step().unwrap();
        assert_eq!(transcript(&out).as_deref(), Some("heard 16000 samples"));
        assert_eq!(out.completed.len(), 1);
        assert!(e.is_idle());
    }

    #[test]
    fn the_audio_survives_the_wire_intact() {
        // base64 of i16 is lossy in the last bit and nothing else; a clip
        // that came back the wrong length would mean a framing bug.
        let mut e = engine(false);
        e.admit(job("r1", 12_345, 16_000)).unwrap();
        assert_eq!(
            transcript(&e.step().unwrap()).as_deref(),
            Some("heard 12345 samples")
        );
    }

    #[test]
    fn a_rate_the_model_does_not_want_is_refused_at_admission() {
        // Not at step time, and not silently: transcribing 44.1 kHz audio as
        // though it were 16 kHz produces fluent text at the wrong speed.
        let mut e = engine(false);
        let err = e.admit(job("r1", 1000, 44_100)).unwrap_err().to_string();
        assert!(err.contains("44100 Hz"), "{err}");
        assert!(err.contains("16000 Hz"), "{err}");
        assert!(e.is_idle());
    }

    #[test]
    fn a_job_of_the_wrong_kind_is_refused() {
        let mut e = engine(false);
        let mut j = job("r1", 1000, 16_000);
        j.kind = JobKind::Chat;
        let err = e.admit(j).unwrap_err().to_string();
        assert!(err.contains("transcribes audio"), "{err}");
    }

    #[test]
    fn a_job_with_no_audio_is_refused() {
        let mut e = engine(false);
        let mut j = job("r1", 1000, 16_000);
        j.audio = None;
        assert!(e.admit(j).is_err());
    }

    #[test]
    fn the_request_is_announced_before_the_long_step() {
        // A clip takes seconds; without this the gateway cannot tell a slow
        // transcription from one still sitting in the queue.
        let mut e = engine(false);
        e.admit(job("r1", 16_000, 16_000)).unwrap();
        let out = e.step().unwrap();
        assert!(matches!(out.events[0].1, Delta::Started { .. }));
    }

    #[test]
    fn a_cancelled_clip_is_never_transcribed() {
        let mut e = engine(false);
        e.admit(job("r1", 1000, 16_000)).unwrap();
        e.admit(job("r2", 1000, 16_000)).unwrap();
        assert!(e.cancel(&RequestId::parse("r1").unwrap()));
        let out = e.step().unwrap();
        assert_eq!(out.completed[0].as_str(), "r2");
    }

    #[test]
    fn a_failed_pass_fails_the_request_rather_than_dropping_it() {
        // The clip must still be in flight after the error, or the caller
        // waits for a timeout instead of being told what went wrong.
        let mut e = engine(true);
        e.admit(job("r1", 1000, 16_000)).unwrap();
        assert!(e.step().is_err());
        assert!(!e.is_idle(), "the request was dropped by the failed step");
        let out = e.fail_all("device wedged");
        assert!(matches!(out.events[0].1, Delta::Failed { .. }));
        assert_eq!(out.completed[0].as_str(), "r1");
        assert!(e.is_idle());
    }

    #[test]
    fn load_is_reported_without_pretending_there_is_a_cache() {
        let mut e = engine(false);
        assert_eq!(e.stats(), (0, 0, 0.0, 0.0));
        e.admit(job("r1", 1000, 16_000)).unwrap();
        e.admit(job("r2", 1000, 16_000)).unwrap();
        let (running, waiting, kv, prefix) = e.stats();
        assert_eq!((running, waiting), (1, 1));
        assert_eq!((kv, prefix), (0.0, 0.0));
    }
}
