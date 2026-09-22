use std::fmt;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("unknown model: {0}")]
    UnknownModel(String),

    #[error("request exceeds context window: {tokens} tokens > {limit}")]
    ContextLengthExceeded { tokens: usize, limit: usize },

    /// The queue ahead of this request is already as long as the operator
    /// allows. Answered with 503 and `Retry-After`, the signal a client's
    /// backoff acts on, rather than 429, which means a per-client quota.
    #[error("engine is at capacity")]
    Overloaded,

    #[error("request cancelled")]
    Cancelled,

    #[error("request timed out after {0:?}")]
    Timeout(std::time::Duration),

    #[error("tokenizer: {0}")]
    Tokenizer(String),

    #[error("chat template: {0}")]
    ChatTemplate(String),

    #[error("transport: {0}")]
    Transport(String),

    #[error("config: {0}")]
    Config(String),

    #[error("engine: {0}")]
    Engine(String),
}

impl Error {
    /// The HTTP status an OpenAI client should see. Keeping this here rather
    /// than in the gateway means a new variant cannot silently default to 500.
    pub fn http_status(&self) -> u16 {
        match self {
            Self::InvalidRequest(_) | Self::ContextLengthExceeded { .. } => 400,
            Self::UnknownModel(_) => 404,
            Self::Overloaded => 503,
            Self::Cancelled => 499,
            Self::Timeout(_) => 504,
            Self::Tokenizer(_)
            | Self::ChatTemplate(_)
            | Self::Transport(_)
            | Self::Config(_)
            | Self::Engine(_) => 500,
        }
    }

    /// OpenAI's `error.type` discriminator.
    pub fn openai_type(&self) -> &'static str {
        match self {
            Self::InvalidRequest(_) | Self::ContextLengthExceeded { .. } => "invalid_request_error",
            Self::UnknownModel(_) => "not_found_error",
            Self::Overloaded => "server_error",
            _ => "api_error",
        }
    }

    /// OpenAI's `error.code`, which clients switch on for retry behaviour.
    pub fn openai_code(&self) -> Option<&'static str> {
        match self {
            Self::ContextLengthExceeded { .. } => Some("context_length_exceeded"),
            Self::UnknownModel(_) => Some("model_not_found"),
            Self::Overloaded => Some("overloaded"),
            _ => None,
        }
    }
}

/// The JSON body shape OpenAI clients parse on a non-2xx response.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl From<&Error> for ErrorBody {
    fn from(e: &Error) -> Self {
        Self {
            error: ErrorDetail {
                message: e.to_string(),
                r#type: e.openai_type().to_string(),
                param: None,
                code: e.openai_code().map(str::to_string),
            },
        }
    }
}

/// Convenience for the many places that produce an `Engine` error from a
/// message.
pub fn engine<T: fmt::Display>(msg: T) -> Error {
    Error::Engine(msg.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_match_openai_conventions() {
        assert_eq!(Error::UnknownModel("x".into()).http_status(), 404);
        assert_eq!(Error::Overloaded.http_status(), 503);
        assert_eq!(
            Error::ContextLengthExceeded {
                tokens: 10,
                limit: 5
            }
            .http_status(),
            400
        );
        assert_eq!(Error::Engine("boom".into()).http_status(), 500);
    }

    #[test]
    fn error_body_carries_code() {
        let e = Error::ContextLengthExceeded {
            tokens: 10,
            limit: 5,
        };
        let body = ErrorBody::from(&e);
        assert_eq!(body.error.code.as_deref(), Some("context_length_exceeded"));
        assert_eq!(body.error.r#type, "invalid_request_error");
    }
}
