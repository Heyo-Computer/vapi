use std::time::Duration;

use futures::StreamExt;
use vapi_core::{Error, FinishReason, RequestId, Result};
use vapi_proto::{Delta, DeltaMsg, DeltaSeqCheck, Subjects};

/// A subscription to one request's token stream, opened *before* the job is
/// published.
///
/// That ordering is not optional. Core NATS has no replay: a delta published
/// before the subscription exists is simply gone. On a prefix-cache hit the
/// first token can come back within single-digit milliseconds, so publishing
/// first and subscribing after loses tokens under exactly the conditions the
/// cache is supposed to make fast.
pub struct TokenStream {
    sub: async_nats::Subscriber,
    seq_check: DeltaSeqCheck,
    request_id: RequestId,
    first_token_timeout: Duration,
    idle_timeout: Duration,
    started: bool,
}

/// What the gateway should do with one received delta.
pub enum StreamEvent {
    Started {
        cached_prefix_tokens: usize,
    },
    Token {
        text: String,
    },
    Done {
        reason: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Failed {
        message: String,
    },
}

impl TokenStream {
    /// Subscribe and flush, so the subscription is registered on the server
    /// before the caller publishes the job.
    pub async fn open(
        client: &async_nats::Client,
        request_id: &RequestId,
        first_token_timeout: Duration,
        idle_timeout: Duration,
    ) -> Result<Self> {
        let sub = client
            .subscribe(Subjects::stream(request_id))
            .await
            .map_err(|e| Error::Transport(format!("subscribe: {e}")))?;
        // Without this flush the SUB may still be buffered client-side when
        // the job publish races ahead of it.
        client
            .flush()
            .await
            .map_err(|e| Error::Transport(format!("flush: {e}")))?;

        Ok(Self {
            sub,
            seq_check: DeltaSeqCheck::default(),
            request_id: request_id.clone(),
            first_token_timeout,
            idle_timeout,
            started: false,
        })
    }

    /// Await the next event, or `None` when the stream has ended.
    pub async fn next(&mut self) -> Option<Result<StreamEvent>> {
        // A request still waiting in the queue gets the longer budget; once a
        // worker is producing, a long gap means it stalled.
        let timeout = if self.started {
            self.idle_timeout
        } else {
            self.first_token_timeout
        };

        let msg = match tokio::time::timeout(timeout, self.sub.next()).await {
            Err(_) => return Some(Err(Error::Timeout(timeout))),
            Ok(None) => return None,
            Ok(Some(m)) => m,
        };

        let parsed: DeltaMsg = match vapi_proto::decode(&msg.payload) {
            Ok(d) => d,
            Err(e) => return Some(Err(e)),
        };

        if let Some(missing) = self.seq_check.observe(parsed.seq) {
            // Core NATS dropped something. Report it rather than serving a
            // completion with a word silently missing.
            metrics::counter!("vapi_stream_gap_total").increment(missing);
            tracing::error!(
                request_id = %self.request_id,
                missing,
                "token stream gap; the response would be incomplete"
            );
            return Some(Err(Error::Transport(format!(
                "{missing} token delta(s) lost in transit"
            ))));
        }

        self.started = true;
        Some(Ok(match parsed.delta {
            Delta::Started {
                cached_prefix_tokens,
                ..
            } => StreamEvent::Started {
                cached_prefix_tokens,
            },
            Delta::Token { text, .. } => StreamEvent::Token { text },
            Delta::Done {
                reason,
                prompt_tokens,
                completion_tokens,
            } => StreamEvent::Done {
                reason,
                prompt_tokens,
                completion_tokens,
            },
            Delta::Failed { message } => StreamEvent::Failed { message },
        }))
    }
}
