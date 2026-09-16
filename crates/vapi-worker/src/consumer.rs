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
    let stream = js
        .get_stream(&cfg.nats.jobs_stream)
        .await
        .map_err(|e| Error::Transport(format!("get stream: {e}")))?;

    let consumer = stream
        .get_or_create_consumer(
            &Subjects::consumer_name(model),
            consumer::pull::Config {
                durable_name: Some(Subjects::consumer_name(model)),
                filter_subject: Subjects::jobs(model),
                ack_policy: consumer::AckPolicy::Explicit,
                ack_wait: cfg.worker.ack_wait(),
                max_deliver: 3,
                max_ack_pending: cfg.worker.max_concurrent_seqs as i64,
                ..Default::default()
            },
        )
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
