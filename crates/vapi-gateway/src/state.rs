use std::sync::Arc;

use vapi_core::{Config, ModelId};
use vapi_tokenize::Tokenization;

use crate::nats::Transport;

pub struct AppState {
    pub cfg: Config,
    pub transport: Transport,
    pub tokenizer: Tokenization,
    pub model: ModelId,
}

pub type SharedState = Arc<AppState>;
