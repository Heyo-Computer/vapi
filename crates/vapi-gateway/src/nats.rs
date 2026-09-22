use std::time::Duration;

use async_nats::jetstream::{self, stream};
use vapi_core::{Config, Error, ModelId, RequestId, Result};
use vapi_proto::{Job, Subjects};

/// NATS connection plus the JetStream context, shared by every request.
#[derive(Clone)]
pub struct Transport {
    pub client: async_nats::Client,
    pub js: jetstream::Context,
    jobs_stream: String,
}

impl Transport {
    pub async fn connect(cfg: &Config) -> Result<Self> {
        let client = async_nats::connect(&cfg.nats.url)
            .await
            .map_err(|e| Error::Transport(format!("connect {}: {e}", cfg.nats.url)))?;
        let js = jetstream::new(client.clone());
        Ok(Self {
            client,
            js,
            jobs_stream: cfg.nats.jobs_stream.clone(),
        })
    }

    /// Create the work-queue stream if it is not already there.
    ///
    /// `WorkQueue` retention is the important setting: a message is removed
    /// once a single consumer acks it, which is what makes several workers
    /// compete for prompts rather than each receiving every one.
    pub async fn ensure_streams(&self, cfg: &Config) -> Result<()> {
        self.js
            .get_or_create_stream(stream::Config {
                name: cfg.nats.jobs_stream.clone(),
                subjects: vec![Subjects::JOBS_WILDCARD.to_string()],
                retention: stream::RetentionPolicy::WorkQueue,
                storage: stream::StorageType::File,
                discard: stream::DiscardPolicy::Old,
                max_age: Duration::from_secs(cfg.nats.job_max_age_secs),
                duplicate_window: Duration::from_secs(cfg.nats.dedupe_window_secs),
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Transport(format!("ensure stream: {e}")))?;
        Ok(())
    }

    /// Publish a job, deduplicated on the request id.
    ///
    /// The `Nats-Msg-Id` header means a gateway retry after an ambiguous
    /// failure cannot enqueue the same prompt twice — the client would
    /// otherwise be billed for, and could receive, two generations.
    pub async fn publish_job(&self, model: &ModelId, job: &Job, partitions: u32) -> Result<()> {
        let payload = vapi_proto::encode(job)?;
        let mut headers = async_nats::HeaderMap::new();
        headers.insert("Nats-Msg-Id", job.request_id.as_str());
        // Routed by the conversation's first block, so a chat's later turns
        // reach the worker that already holds its earlier ones.
        let subject = Subjects::jobs_for(
            model,
            &job.prompt_tokens,
            vapi_cache::BLOCK_SIZE,
            partitions,
        );

        let ack = self
            .js
            .publish_with_headers(subject, headers, payload)
            .await
            .map_err(|e| Error::Transport(format!("publish: {e}")))?;
        ack.await
            .map_err(|e| Error::Transport(format!("publish ack: {e}")))?;
        Ok(())
    }

    /// Tell any worker holding this request to stop.
    pub async fn cancel(&self, request_id: &RequestId) {
        // Best effort: the client has already gone, so there is nobody to
        // report a failure to. Worst case the worker finishes and discards.
        let _ = self
            .client
            .publish(Subjects::cancel(request_id), "".into())
            .await;
        let _ = self.client.flush().await;
    }

    pub fn jobs_stream(&self) -> &str {
        &self.jobs_stream
    }
}
