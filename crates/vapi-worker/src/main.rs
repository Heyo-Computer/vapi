mod consumer;
mod engine;
mod registry;
mod runner;

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Duration;

use async_nats::jetstream::{self, AckKind};
use futures::StreamExt;
use metrics_exporter_prometheus::PrometheusBuilder;
use vapi_core::{Config, ModelId, RequestId};
use vapi_engine::MockBackend;
use vapi_engine::backend::ExecutionBackend;
use vapi_proto::{Job, Subjects};
use vapi_tokenize::Tokenization;

use crate::engine::Engine;
use crate::runner::{Command, Stats};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,vapi_worker=debug".into()),
        )
        .init();

    let cfg = Config::load_or_default("vapi.toml")?;
    let model = ModelId(cfg.model.id.clone());
    let worker_id = cfg
        .worker
        .id
        .clone()
        .unwrap_or_else(|| format!("w-{}", uuid_like()));

    if let Some(bind) = &cfg.worker.metrics_bind {
        let addr: std::net::SocketAddr = bind.parse()?;
        PrometheusBuilder::new()
            .with_http_listener(addr)
            .install()?;
        tracing::info!(%bind, "metrics listening");
    }

    let tokenizer = match &cfg.model.path {
        Some(dir) => Tokenization::from_dir(dir)?,
        None => {
            tracing::warn!("no model.path configured; using the byte-level tokenizer");
            Tokenization::bytes()?
        }
    };

    let backend = build_backend(&cfg, &tokenizer)?;
    let engine = Engine::new(&cfg, backend, tokenizer, worker_id.clone());

    let client = async_nats::connect(&cfg.nats.url).await?;
    let js = jetstream::new(client.clone());
    consumer::wait_for_stream(&js, &cfg.nats.jobs_stream).await?;
    let consumers = consumer::ensure_consumers(&js, &cfg, &model).await?;
    let partitions: Vec<u32> = consumers.iter().map(|(p, _)| *p).collect();

    // The engine runs on its own thread so a long forward pass never holds
    // up cancels or ack heartbeats.
    let mut handle = runner::spawn(engine, cfg.worker.max_step_failures);

    // One stream per partition, merged: the worker does not care which
    // queue a prompt came from once it has it.
    let mut streams = Vec::with_capacity(consumers.len());
    for (_, c) in &consumers {
        streams.push(c.messages().await?);
    }
    let mut jobs = futures::stream::select_all(streams);
    let mut cancels = client.subscribe(Subjects::CANCEL_WILDCARD).await?;
    // Keeps long generations from being redelivered: JetStream would
    // otherwise assume the worker died and hand the prompt to someone else,
    // duplicating the user's completion.
    let mut heartbeat = tokio::time::interval(cfg.worker.ack_progress_interval());
    let mut inflight: HashMap<RequestId, jetstream::Message> = HashMap::new();
    let mut stats = Stats::default();

    // The registry is observability, not routing: a worker that fails to
    // publish itself still serves its partitions.
    let registry = registry::Registry::open(&js, &cfg).await;
    if registry.is_none() {
        tracing::warn!("worker registry unavailable; continuing without it");
    }
    let started = std::time::Instant::now();

    tracing::info!(
        %worker_id,
        model = %model,
        ?partitions,
        of = cfg.nats.job_partitions.max(1),
        "worker ready"
    );

    // Graceful drain: on SIGTERM or Ctrl-C stop taking jobs, let what is
    // running finish, and fail whatever is left when the deadline passes.
    // Cancels and heartbeats keep flowing meanwhile so JetStream does not
    // redeliver a prompt that is still being served here.
    let mut shutdown = std::pin::pin!(shutdown_signal());
    let mut draining = false;
    let drain_deadline = tokio::time::sleep(std::time::Duration::MAX);
    let mut drain_deadline = std::pin::pin!(drain_deadline);

    loop {
        // Take new work only while there is room; `max_ack_pending` is the
        // real limit, this just avoids buffering past it.
        let want_work = !draining && handle.headroom.load(Ordering::Relaxed) > 0;
        if draining && inflight.is_empty() {
            tracing::info!("drained; exiting");
            if let Some(r) = &registry {
                r.remove(&worker_id).await;
            }
            break;
        }

        tokio::select! {
            biased;

            _ = &mut shutdown, if !draining => {
                draining = true;
                let timeout = std::time::Duration::from_secs(cfg.worker.drain_timeout_secs);
                tracing::info!(inflight = inflight.len(), ?timeout, "draining");
                drain_deadline.as_mut().reset(tokio::time::Instant::now() + timeout);
            }

            _ = &mut drain_deadline, if draining => {
                tracing::warn!(inflight = inflight.len(), "drain timeout; failing what is left");
                let _ = handle.commands.send(Command::FailAll("worker shutting down".into()));
                // Never fires again.
                drain_deadline.as_mut().reset(tokio::time::Instant::now() + std::time::Duration::from_secs(86_400));
            }

            Some(msg) = cancels.next() => {
                if let Some(id) = msg.subject.as_str().rsplit('.').next()
                    && let Some(rid) = RequestId::parse(id)
                {
                    let _ = handle.commands.send(Command::Cancel(rid.clone()));
                    // Ack rather than nak: the work is done with, and a nak
                    // would hand the abandoned prompt to another worker.
                    if let Some(m) = inflight.remove(&rid) {
                        let _ = m.ack().await;
                    }
                }
            }

            Some(Ok(msg)) = jobs.next(), if want_work => {
                match vapi_proto::decode::<Job>(&msg.payload) {
                    Ok(job) => {
                        let rid = job.request_id.clone();
                        metrics::counter!("vapi_jobs_admitted_total").increment(1);
                        inflight.insert(rid, msg);
                        if handle.commands.send(Command::Admit(Box::new(job))).is_err() {
                            anyhow::bail!("engine thread is gone");
                        }
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "undecodable job; terminating it");
                        // Malformed and will never decode, so redelivery is
                        // pointless — Term stops it permanently.
                        let _ = msg.ack_with(AckKind::Term).await;
                    }
                }
            }

            _ = heartbeat.tick() => {
                for m in inflight.values() {
                    let _ = m.ack_with(AckKind::Progress).await;
                }
                record_gauges(&stats);
                if let Some(r) = &registry {
                    r.publish(vapi_proto::WorkerStats {
                        worker_id: worker_id.clone(),
                        model: model.clone(),
                        running: stats.running,
                        waiting: stats.waiting,
                        max_concurrent: cfg.worker.max_concurrent_seqs,
                        block_utilization: stats.kv_utilization,
                        prefix_hit_rate: stats.prefix_hit_rate,
                        uptime_secs: started.elapsed().as_secs(),
                    }, &partitions).await;
                }
            }

            report = handle.reports.recv() => {
                let Some(report) = report else {
                    anyhow::bail!("engine thread exited");
                };
                for (rid, msg) in report.deltas {
                    publish(&client, &rid, msg).await;
                }
                for rid in report.completed {
                    if let Some(m) = inflight.remove(&rid) {
                        // Ack only now: the prompt has been fully served.
                        if let Err(e) = m.ack().await {
                            tracing::error!(request_id = %rid, error = %e, "ack failed");
                        }
                    }
                }
                stats = report.stats;
                record_gauges(&stats);
            }
        }
    }
    Ok(())
}

/// SIGTERM (what an orchestrator sends) or Ctrl-C.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// The real model when there is one and the binary can run it; the mock
/// otherwise, so the full request path still works with nothing downloaded.
fn build_backend(
    cfg: &Config,
    tokenizer: &Tokenization,
) -> anyhow::Result<Box<dyn ExecutionBackend>> {
    #[cfg(feature = "candle")]
    if let Some(dir) = &cfg.model.path {
        let opts = vapi_backend_candle::LoadOptions {
            dtype: cfg.model.dtype,
            device: cfg.model.device,
            num_blocks: cfg.model.num_blocks,
            kv_cache_fraction: cfg.model.kv_cache_fraction,
            cuda_graphs: cfg.model.cuda_graphs,
        };
        tracing::info!(dir = %dir.display(), "loading model");
        let backend = vapi_backend_candle::CandleBackend::load(dir, &opts)?;
        tracing::info!(spec = ?backend.spec(), "model loaded");
        return Ok(Box::new(backend));
    }

    #[cfg(not(feature = "candle"))]
    if cfg.model.path.is_some() {
        tracing::warn!(
            "model.path is set but this binary was built without the `candle` feature; \
             using the mock backend with the real tokenizer"
        );
    }

    let num_blocks = cfg.model.num_blocks.unwrap_or(2048);
    let mut spec = vapi_engine::ModelSpec::tiny();
    spec.vocab_size = tokenizer.vocab_hint().max(spec.vocab_size);
    spec.max_context = cfg.model.max_context;
    spec.eos_token_ids = tokenizer.eos_token_ids();
    tracing::warn!("running the MOCK backend; output is not from a real model");
    Ok(Box::new(
        MockBackend::new(num_blocks, vapi_cache::BLOCK_SIZE)
            .with_spec(spec)
            .with_step_delay(Duration::from_millis(cfg.worker.mock_step_delay_ms)),
    ))
}

fn record_gauges(s: &Stats) {
    metrics::gauge!("vapi_scheduler_running").set(s.running as f64);
    metrics::gauge!("vapi_scheduler_waiting").set(s.waiting as f64);
    metrics::gauge!("vapi_kv_cache_utilization").set(s.kv_utilization as f64);
    metrics::gauge!("vapi_prefix_cache_hit_rate").set(s.prefix_hit_rate as f64);
}

async fn publish(client: &async_nats::Client, rid: &RequestId, msg: vapi_proto::DeltaMsg) {
    match vapi_proto::encode(&msg) {
        Ok(payload) => {
            if let Err(e) = client.publish(Subjects::stream(rid), payload).await {
                tracing::warn!(request_id = %rid, error = %e, "delta publish failed");
            }
        }
        Err(e) => tracing::error!(request_id = %rid, error = %e, "delta encode failed"),
    }
}

fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{n:08x}")
}
