//! The engine on its own OS thread.
//!
//! `Engine::step` is CPU-bound (or, on a GPU, blocks on the device), and a
//! forward pass over a full batch can take tens of milliseconds. Running it
//! inline in the async loop meant every cancel and every JetStream progress
//! ack waited for the current step. Here the engine owns a thread and talks
//! to the async side over channels: commands in, step results out.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use tokio::sync::mpsc;
use vapi_core::RequestId;
use vapi_proto::{Delta, DeltaMsg, Job};

use crate::decision::DecisionEngine;
use crate::engine::{Engine, StepOutput};

/// What the worker's thread needs of an engine.
///
/// Two kinds implement it: the continuous-batching decoder and the
/// single-pass decision engine. A worker runs one or the other for its whole
/// life, decided by the model it loaded — a checkpoint is either a decoder or
/// an encoder, never both — so this is a choice made once at startup rather
/// than per job.
pub trait WorkerEngine: Send + 'static {
    fn admit(&mut self, job: Job) -> vapi_core::Result<()>;
    fn cancel(&mut self, request_id: &RequestId) -> bool;
    fn fail_all(&mut self, message: &str) -> StepOutput;
    fn step(&mut self) -> vapi_core::Result<StepOutput>;
    fn is_idle(&self) -> bool;
    fn headroom(&self) -> usize;
    fn stats(&self) -> (usize, usize, f32, f32);
    fn sequence(&mut self, request_id: &RequestId, delta: Delta) -> DeltaMsg;
    fn forget(&mut self, request_id: &RequestId);
}

impl WorkerEngine for Engine {
    fn admit(&mut self, job: Job) -> vapi_core::Result<()> {
        Engine::admit(self, job).map(|_| ())
    }
    fn cancel(&mut self, request_id: &RequestId) -> bool {
        Engine::cancel(self, request_id)
    }
    fn fail_all(&mut self, message: &str) -> StepOutput {
        Engine::fail_all(self, message)
    }
    fn step(&mut self) -> vapi_core::Result<StepOutput> {
        Engine::step(self)
    }
    fn is_idle(&self) -> bool {
        Engine::is_idle(self)
    }
    fn headroom(&self) -> usize {
        Engine::headroom(self)
    }
    fn stats(&self) -> (usize, usize, f32, f32) {
        Engine::stats(self)
    }
    fn sequence(&mut self, request_id: &RequestId, delta: Delta) -> DeltaMsg {
        Engine::sequence(self, request_id, delta)
    }
    fn forget(&mut self, request_id: &RequestId) {
        Engine::forget(self, request_id)
    }
}

impl WorkerEngine for DecisionEngine {
    fn admit(&mut self, job: Job) -> vapi_core::Result<()> {
        DecisionEngine::admit(self, job)
    }
    fn cancel(&mut self, request_id: &RequestId) -> bool {
        DecisionEngine::cancel(self, request_id)
    }
    fn fail_all(&mut self, message: &str) -> StepOutput {
        DecisionEngine::fail_all(self, message)
    }
    fn step(&mut self) -> vapi_core::Result<StepOutput> {
        DecisionEngine::step(self)
    }
    fn is_idle(&self) -> bool {
        DecisionEngine::is_idle(self)
    }
    fn headroom(&self) -> usize {
        DecisionEngine::headroom(self)
    }
    fn stats(&self) -> (usize, usize, f32, f32) {
        DecisionEngine::stats(self)
    }
    fn sequence(&mut self, request_id: &RequestId, delta: Delta) -> DeltaMsg {
        DecisionEngine::sequence(self, request_id, delta)
    }
    fn forget(&mut self, request_id: &RequestId) {
        DecisionEngine::forget(self, request_id)
    }
}

pub enum Command {
    Admit(Box<Job>),
    Cancel(RequestId),
    /// Fail everything in flight with this message; a drain that ran out
    /// of time.
    FailAll(String),
}

/// Snapshot of engine load, for the metrics gauges.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub running: usize,
    pub waiting: usize,
    pub kv_utilization: f32,
    pub prefix_hit_rate: f32,
}

/// What the engine thread reports after processing commands or a step.
#[derive(Default)]
pub struct Report {
    /// Deltas to publish, already stamped with their stream sequence.
    pub deltas: Vec<(RequestId, DeltaMsg)>,
    /// Requests whose JetStream message can now be acked, terminal event
    /// included in `deltas`.
    pub completed: Vec<RequestId>,
    pub stats: Stats,
}

pub struct EngineHandle {
    pub commands: mpsc::UnboundedSender<Command>,
    pub reports: mpsc::UnboundedReceiver<Report>,
    /// Sequences the engine can still admit, refreshed by the engine thread.
    /// Slightly stale by construction; JetStream's `max_ack_pending` is the
    /// hard limit.
    pub headroom: Arc<AtomicUsize>,
}

/// Move `engine` onto a dedicated thread and return the channels to it.
///
/// A step that fails fails every request in flight (see
/// [`Engine::fail_all`]) and the loop continues; after `max_step_failures`
/// failures in a row the thread exits, which makes the worker process exit,
/// which is what gets a wedged device restarted.
pub fn spawn<E: WorkerEngine>(mut engine: E, max_step_failures: usize) -> EngineHandle {
    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<Command>();
    let (rep_tx, rep_rx) = mpsc::unbounded_channel::<Report>();
    let headroom = Arc::new(AtomicUsize::new(engine.headroom()));
    let headroom_w = headroom.clone();

    thread::Builder::new()
        .name("vapi-engine".into())
        .spawn(move || {
            let mut failures_in_a_row = 0usize;
            loop {
                let mut report = Report::default();

                // Idle: block until there is something to do. Busy: take
                // whatever has arrived, then step.
                if engine.is_idle() {
                    match cmd_rx.blocking_recv() {
                        Some(cmd) => handle(&mut engine, cmd, &mut report),
                        None => break,
                    }
                }
                while let Ok(cmd) = cmd_rx.try_recv() {
                    handle(&mut engine, cmd, &mut report);
                }

                if !engine.is_idle() {
                    match engine.step() {
                        Ok(out) => {
                            failures_in_a_row = 0;
                            for (rid, delta) in out.events {
                                let msg = engine.sequence(&rid, delta);
                                report.deltas.push((rid, msg));
                            }
                            for rid in out.completed {
                                engine.forget(&rid);
                                report.completed.push(rid);
                            }
                        }
                        Err(e) => {
                            failures_in_a_row += 1;
                            metrics::counter!("vapi_engine_step_failures_total").increment(1);
                            tracing::error!(
                                error = %e,
                                failures_in_a_row,
                                "engine step failed; failing every request in flight"
                            );
                            let out = engine.fail_all(&format!("engine step failed: {e}"));
                            for (rid, delta) in out.events {
                                let msg = engine.sequence(&rid, delta);
                                report.deltas.push((rid, msg));
                            }
                            for rid in out.completed {
                                engine.forget(&rid);
                                report.completed.push(rid);
                            }
                            if max_step_failures > 0 && failures_in_a_row >= max_step_failures {
                                tracing::error!(
                                    max_step_failures,
                                    "too many consecutive step failures; exiting"
                                );
                                let _ = rep_tx.send(report);
                                break;
                            }
                        }
                    }
                }

                headroom_w.store(engine.headroom(), Ordering::Relaxed);
                let (running, waiting, kv_utilization, prefix_hit_rate) = engine.stats();
                report.stats = Stats {
                    running,
                    waiting,
                    kv_utilization,
                    prefix_hit_rate,
                };
                if rep_tx.send(report).is_err() {
                    break;
                }
            }
            tracing::info!("engine thread exiting");
        })
        .expect("spawn engine thread");

    EngineHandle {
        commands: cmd_tx,
        reports: rep_rx,
        headroom,
    }
}

fn handle<E: WorkerEngine>(engine: &mut E, cmd: Command, report: &mut Report) {
    match cmd {
        Command::Admit(job) => {
            let rid = job.request_id.clone();
            if let Err(e) = engine.admit(*job) {
                tracing::warn!(request_id = %rid, error = %e, "rejected");
                let msg = engine.sequence(
                    &rid,
                    Delta::Failed {
                        message: e.to_string(),
                    },
                );
                engine.forget(&rid);
                report.deltas.push((rid.clone(), msg));
                report.completed.push(rid);
            }
        }
        Command::Cancel(rid) => {
            if engine.cancel(&rid) {
                tracing::debug!(request_id = %rid, "cancelled");
            }
            engine.forget(&rid);
        }
        Command::FailAll(message) => {
            let out = engine.fail_all(&message);
            for (rid, delta) in out.events {
                let msg = engine.sequence(&rid, delta);
                report.deltas.push((rid, msg));
            }
            for rid in out.completed {
                engine.forget(&rid);
                report.completed.push(rid);
            }
        }
    }
}
