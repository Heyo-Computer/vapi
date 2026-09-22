use std::time::Duration;

use async_nats::jetstream::{self, consumer, stream};
use vapi_core::{Config, Error, ModelId, Result};
use vapi_proto::Subjects;

/// Create (or attach to) the durable pull consumer this worker pool shares.
///
/// All workers for a model use one consumer name, which is what makes them
/// compete for prompts instead of each receiving every one.
///
/// `max_ack_pending` is the backpressure mechanism, and the most important
/// setting here: NATS stops delivering once this many messages are unacked, so
/// a burst queues in the stream rather than in the worker's memory.
pub async fn ensure_consumer(
    js: &jetstream::Context,
    cfg: &Config,
    model: &ModelId,
) -> Result<consumer::PullConsumer> {
    ensure_partition_consumer(js, cfg, model, None).await
}

/// One consumer per partition this worker serves.
///
/// With affinity routing on, each partition is a queue of its own: a worker
/// that consumed them all would defeat the routing, and one durable
/// consumer cannot filter to a subset of subjects across NATS versions. So
/// there is a consumer per partition, named after it, and the worker merges
/// their streams.
pub async fn ensure_consumers(
    js: &jetstream::Context,
    cfg: &Config,
    model: &ModelId,
) -> Result<Vec<(u32, consumer::PullConsumer)>> {
    let partitions = cfg.nats.job_partitions.max(1);
    if partitions == 1 {
        return Ok(vec![(0, ensure_consumer(js, cfg, model).await?)]);
    }
    let owned = cfg.worker.owned_partitions(partitions);
    if owned.is_empty() {
        return Err(Error::Config(format!(
            "worker.partitions selects nothing out of {partitions} partitions"
        )));
    }
    let mut out = Vec::with_capacity(owned.len());
    for p in owned {
        out.push((p, ensure_partition_consumer(js, cfg, model, Some(p)).await?));
    }
    Ok(out)
}

async fn ensure_partition_consumer(
    js: &jetstream::Context,
    cfg: &Config,
    model: &ModelId,
    partition: Option<u32>,
) -> Result<consumer::PullConsumer> {
    let stream = js
        .get_stream(&cfg.nats.jobs_stream)
        .await
        .map_err(|e| Error::Transport(format!("get stream: {e}")))?;

    // `create_consumer`, not `get_or_create_consumer`: the consumer is durable
    // and survives restarts, and get-or-create keeps whatever config it was
    // first made with. That silently pinned `max_ack_pending` to a stale
    // value once, capping the engine at 8 sequences while the config said 64.
    // The create API updates the updatable fields of an existing consumer
    // (max_ack_pending and ack_wait among them) and errors on the rest.
    let (name, subject) = match partition {
        None => (Subjects::consumer_name(model), Subjects::jobs(model)),
        Some(p) => (
            format!("{}-p{p}", Subjects::consumer_name(model)),
            Subjects::jobs_partition(model, p),
        ),
    };
    let consumer = stream
        .create_consumer(consumer::pull::Config {
            durable_name: Some(name),
            filter_subject: subject,
            ack_policy: consumer::AckPolicy::Explicit,
            ack_wait: cfg.worker.ack_wait(),
            max_deliver: 3,
            max_ack_pending: cfg.worker.max_concurrent_seqs as i64,
            ..Default::default()
        })
        .await
        .map_err(|e| Error::Transport(format!("create consumer: {e}")))?;

    Ok(consumer)
}

/// Stream config is created by the gateway; this verifies it exists so a
/// worker started first fails with a clear message.
pub async fn wait_for_stream(js: &jetstream::Context, name: &str) -> Result<stream::Stream> {
    for attempt in 0..30 {
        match js.get_stream(name).await {
            Ok(s) => return Ok(s),
            Err(e) if attempt == 29 => {
                return Err(Error::Transport(format!(
                    "stream {name} never appeared: {e}"
                )));
            }
            Err(_) => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    }
    unreachable!()
}
