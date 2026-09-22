use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use vapi_core::{Config, ModelId};
use vapi_tokenize::Tokenization;

use crate::nats::Transport;

pub struct AppState {
    pub cfg: Config,
    pub transport: Transport,
    pub tokenizer: Tokenization,
    pub model: ModelId,
    /// Requests published but not yet started by a worker; see
    /// [`AppState::queue_slot`].
    pub queued: AtomicUsize,
    /// Tier 2, when `cache.response_cache` is on.
    pub response_cache: Option<crate::response_cache::ResponseCache>,
    /// Weight fingerprint of the served model, part of every cache key.
    pub fingerprint: String,
    /// The model's structured-output markers, when it has them.
    pub output_format: Option<crate::output::OutputFormat>,
}

pub type SharedState = Arc<AppState>;

impl AppState {
    /// Take a place in the queue, or refuse with `Overloaded` when
    /// `gateway.max_queued_requests` are already waiting. The slot is given
    /// back when a worker starts the request, or when the request ends
    /// without ever starting.
    pub fn queue_slot(self: &Arc<Self>) -> Result<QueueSlot, vapi_core::Error> {
        let max = self.cfg.gateway.max_queued_requests;
        let mut cur = self.queued.load(Ordering::Relaxed);
        loop {
            if max > 0 && cur >= max {
                metrics::counter!("vapi_gateway_overloaded_total").increment(1);
                return Err(vapi_core::Error::Overloaded);
            }
            match self.queued.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(now) => cur = now,
            }
        }
        metrics::gauge!("vapi_gateway_queued").set((cur + 1) as f64);
        Ok(QueueSlot {
            state: Some(self.clone()),
        })
    }
}

/// A place in the gateway's queue; released on drop or [`QueueSlot::release`].
pub struct QueueSlot {
    state: Option<Arc<AppState>>,
}

impl QueueSlot {
    pub fn release(&mut self) {
        if let Some(st) = self.state.take() {
            let now = st.queued.fetch_sub(1, Ordering::Relaxed) - 1;
            metrics::gauge!("vapi_gateway_queued").set(now as f64);
        }
    }
}

impl Drop for QueueSlot {
    fn drop(&mut self) {
        self.release();
    }
}
