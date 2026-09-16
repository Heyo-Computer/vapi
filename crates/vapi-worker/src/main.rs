mod consumer;
mod engine;

use std::collections::HashMap;
use std::time::Duration;

use async_nats::jetstream::{self, AckKind};
use futures::StreamExt;
use metrics_exporter_prometheus::PrometheusBuilder;
use vapi_core::{Config, ModelId, RequestId};
use vapi_engine::MockBackend;
use vapi_proto::{Job, Subjects};
use vapi_tokenize::Tokenization;

use crate::engine::Engine;

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

    // M1 runs on the mock backend: the whole request path — queueing,
    // scheduling, paged caching, streaming — is exercised without weights.
    // Swapping in the candle backend is a one-line change here.
    let num_blocks = cfg.model.num_blocks.unwrap_or(2048);
    let mut spec = vapi_engine::ModelSpec::tiny();
    spec.vocab_size = tokenizer.vocab_hint().max(spec.vocab_size);
    spec.max_context = cfg.model.max_context;
    spec.eos_token_ids = tokenizer.eos_token_ids();
    let backend = Box::new(
        MockBackend::new(num_blocks, vapi_cache::BLOCK_SIZE)
            .with_spec(spec)
            .with_step_delay(Duration::from_millis(cfg.worker.mock_step_delay_ms)),
    );
    tracing::warn!("running the MOCK backend; output is not from a real model");

    let client = async_nats::connect(&cfg.nats.url).await?;
    let js = jetstream::new(client.clone());
    consumer::wait_for_stream(&js, &cfg.nats.jobs_stream).await?;
    let pull = consumer::ensure_consumer(&js, &cfg, &model).await?;

    let mut engine = Engine::new(&cfg, backend, tokenizer, worker_id.clone());

    let mut jobs = pull.messages().await?;
    let mut cancels = client.subscribe(Subjects::CANCEL_WILDCARD).await?;
    // Keeps long generations from being redelivered: JetStream would
    // otherwise assume the worker died and hand the prompt to someone else,
    // duplicating the user's completion.
    let mut heartbeat = tokio::time::interval(cfg.worker.ack_progress_interval());
    let mut inflight: HashMap<RequestId, jetstream::Message> = HashMap::new();

    tracing::info!(%worker_id, model = %model, "worker ready");

    loop {
        // Take new work only while there is room; `max_ack_pending` is the
        // real limit, this just avoids buffering past it.
        let want_work = engine.headroom() > 0;

        tokio::select! {
            biased;

            Some(msg) = cancels.next() => {
                if let Some(id) = msg.subject.as_str().rsplit('.').next()
                    && let Some(rid) = RequestId::parse(id)
                {
                    if engine.cancel(&rid) {
                        tracing::debug!(request_id = %rid, "cancelled");
                    }
                    // Ack rather than nak: the work is done with, and a nak
                    // would hand the abandoned prompt to another worker.
                    if let Some(m) = inflight.remove(&rid) {
                        let _ = m.ack().await;
                    }
                    engine.forget(&rid);
                }
            }

            Some(Ok(msg)) = jobs.next(), if want_work => {
                match vapi_proto::decode::<Job>(&msg.payload) {
                    Ok(job) => {
                        let rid = job.request_id.clone();
                        match engine.admit(job) {
                            Ok(_) => { inflight.insert(rid, msg); }
                            Err(e) => {
                                tracing::warn!(request_id = %rid, error = %e, "rejected");
                                publish(&client, &mut engine, &rid,
                                    vapi_proto::Delta::Failed { message: e.to_string() }).await;
                                let _ = msg.ack().await;
                            }
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
                record_gauges(&engine);
            }

            _ = tokio::task::yield_now(), if !engine.is_idle() => {}
        }

        if engine.is_idle() {
            continue;
        }

        match engine.step() {
            Ok(out) => {
                for (rid, delta) in out.events {
                    publish(&client, &mut engine, &rid, delta).await;
                }
                record_gauges(&engine);
                for rid in out.completed {
                    if let Some(m) = inflight.remove(&rid) {
                        // Ack only now: the prompt has been fully served.
                        if let Err(e) = m.ack().await {
                            tracing::error!(request_id = %rid, error = %e, "ack failed");
                        }
                    }
                    engine.forget(&rid);
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "engine step failed");
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
}

fn record_gauges(engine: &Engine) {
    let (running, waiting, util, hit) = engine.stats();
    metrics::gauge!("vapi_scheduler_running").set(running as f64);
    metrics::gauge!("vapi_scheduler_waiting").set(waiting as f64);
    metrics::gauge!("vapi_kv_cache_utilization").set(util as f64);
    metrics::gauge!("vapi_prefix_cache_hit_rate").set(hit as f64);
}

async fn publish(
    client: &async_nats::Client,
    engine: &mut Engine,
    rid: &RequestId,
    delta: vapi_proto::Delta,
) {
    let msg = engine.sequence(rid, delta);
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
