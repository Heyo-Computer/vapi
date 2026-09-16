use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response, Sse, sse::Event};
use futures::stream::Stream;
use vapi_core::{Error, FinishReason, RequestId};
use vapi_openai::{
    ChatCompletionRequest, ChatCompletionResponse, CompletionRequest, CompletionResponse, Model,
    ModelList, StreamChunk, Usage,
};
use vapi_proto::{Job, JobKind, Subjects};

use crate::state::SharedState;
use crate::stream::{StreamEvent, TokenStream};

/// Wraps a `vapi_core::Error` into the JSON body OpenAI clients parse.
pub struct ApiError(pub Error);

impl From<Error> for ApiError {
    fn from(e: Error) -> Self {
        Self(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.0.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = vapi_core::error::ErrorBody::from(&self.0);
        (status, Json(body)).into_response()
    }
}

type ApiResult<T> = std::result::Result<T, ApiError>;

pub async fn health() -> &'static str {
    "ok"
}

pub async fn list_models(State(st): State<SharedState>) -> Json<ModelList> {
    Json(ModelList::new(vec![Model::new(st.model.0.clone())]))
}

/// Everything a queued request needs, assembled once and shared by the
/// streaming and non-streaming paths.
struct Prepared {
    request_id: RequestId,
    job: Job,
    created: u64,
    prompt_tokens: usize,
}

fn prepare(
    st: &SharedState,
    kind: JobKind,
    prompt_tokens: Vec<u32>,
    params: vapi_core::SamplingParams,
) -> ApiResult<Prepared> {
    let request_id = RequestId::new();
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if prompt_tokens.len() >= st.cfg.model.max_context {
        return Err(Error::ContextLengthExceeded {
            tokens: prompt_tokens.len(),
            limit: st.cfg.model.max_context,
        }
        .into());
    }

    let n = prompt_tokens.len();
    Ok(Prepared {
        job: Job {
            request_id: request_id.clone(),
            model: st.model.clone(),
            kind,
            prompt_tokens,
            params,
            namespace: st.cfg.cache.default_namespace.clone(),
            reply_to: Subjects::stream(&request_id),
            enqueued_at_ms: created * 1000,
        },
        request_id,
        created,
        prompt_tokens: n,
    })
}

pub async fn chat_completions(
    State(st): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> ApiResult<Response> {
    let params = req
        .to_sampling_params(st.cfg.model.max_context / 4)
        .map_err(|e| ApiError(Error::InvalidRequest(e)))?;
    let tokens = st.tokenizer.encode_chat(&req.messages)?;
    let prepared = prepare(&st, JobKind::Chat, tokens, params)?;

    let streaming = req.is_streaming();
    let include_usage = req.include_usage();
    let model_name = req.model.clone();

    if streaming {
        let id = format!("chatcmpl-{}", prepared.request_id);
        Ok(sse_response(st, prepared, id, model_name, include_usage).await?)
    } else {
        let id = format!("chatcmpl-{}", prepared.request_id);
        let (text, reason, usage) = collect(st, prepared).await?;
        Ok(Json(ChatCompletionResponse::new(
            id, model_name, text, reason, usage,
        ))
        .into_response())
    }
}

pub async fn completions(
    State(st): State<SharedState>,
    Json(req): Json<CompletionRequest>,
) -> ApiResult<Response> {
    let params = req
        .to_sampling_params(st.cfg.model.max_context / 4)
        .map_err(|e| ApiError(Error::InvalidRequest(e)))?;
    let tokens = st.tokenizer.encode(&req.prompt)?;
    let prepared = prepare(&st, JobKind::Completion, tokens, params)?;
    let id = format!("cmpl-{}", prepared.request_id);
    let model_name = req.model.clone();

    if req.is_streaming() {
        Ok(sse_response(st, prepared, id, model_name, false).await?)
    } else {
        let (text, reason, usage) = collect(st, prepared).await?;
        Ok(Json(CompletionResponse::new(id, model_name, text, reason, usage)).into_response())
    }
}

/// Subscribe, publish, and drain the whole stream into one response body.
async fn collect(st: SharedState, prepared: Prepared) -> ApiResult<(String, FinishReason, Usage)> {
    let mut stream = open_and_publish(&st, &prepared).await?;
    let mut text = String::new();
    let mut finish = FinishReason::Stop;
    let mut usage = Usage::new(prepared.prompt_tokens, 0);

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Started {
                cached_prefix_tokens,
            } => {
                metrics::counter!("vapi_cached_prefix_tokens_total")
                    .increment(cached_prefix_tokens as u64);
            }
            StreamEvent::Token { text: t } => text.push_str(&t),
            StreamEvent::Done {
                reason,
                prompt_tokens,
                completion_tokens,
            } => {
                finish = reason;
                usage = Usage::new(prompt_tokens, completion_tokens);
                break;
            }
            StreamEvent::Failed { message } => return Err(Error::Engine(message).into()),
        }
    }
    Ok((text, finish, usage))
}

/// Open the token stream first, then publish the job.
async fn open_and_publish(st: &SharedState, prepared: &Prepared) -> ApiResult<TokenStream> {
    let stream = TokenStream::open(
        &st.transport.client,
        &prepared.request_id,
        Duration::from_secs(st.cfg.gateway.first_token_timeout_secs),
        Duration::from_secs(st.cfg.gateway.stream_idle_timeout_secs),
    )
    .await?;
    st.transport.publish_job(&st.model, &prepared.job).await?;
    Ok(stream)
}

async fn sse_response(
    st: SharedState,
    prepared: Prepared,
    id: String,
    model_name: String,
    include_usage: bool,
) -> ApiResult<Response> {
    let stream = open_and_publish(&st, &prepared).await?;
    let body = sse_stream(st, stream, prepared, id, model_name, include_usage);
    Ok(Sse::new(body).into_response())
}

fn sse_stream(
    st: SharedState,
    mut stream: TokenStream,
    prepared: Prepared,
    id: String,
    model_name: String,
    include_usage: bool,
) -> impl Stream<Item = std::result::Result<Event, std::convert::Infallible>> {
    async_stream::stream! {
        // If the client disconnects, axum drops this stream mid-poll and the
        // guard's Drop tells the worker to stop. Without it a closed browser
        // tab keeps a GPU busy generating tokens nobody will read.
        let _cancel = CancelOnDrop {
            transport: st.transport.clone(),
            request_id: prepared.request_id.clone(),
            finished: false,
        };
        let mut guard = _cancel;

        yield Ok(Event::default().data(
            serde_json::to_string(&StreamChunk::role(&id, &model_name, prepared.created))
                .unwrap_or_default(),
        ));

        let mut completion_tokens = 0usize;
        let mut prompt_tokens = prepared.prompt_tokens;

        while let Some(event) = stream.next().await {
            match event {
                Ok(StreamEvent::Started { cached_prefix_tokens }) => {
                    // Worth surfacing: it is the per-request view of whether
                    // the shared prefix cache is doing its job.
                    metrics::counter!("vapi_cached_prefix_tokens_total")
                        .increment(cached_prefix_tokens as u64);
                    tracing::debug!(
                        request_id = %prepared.request_id,
                        cached_prefix_tokens,
                        "generation started"
                    );
                }
                Ok(StreamEvent::Token { text }) => {
                    completion_tokens += 1;
                    let chunk = StreamChunk::content(&id, &model_name, prepared.created, text);
                    yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                }
                Ok(StreamEvent::Done { reason, prompt_tokens: p, completion_tokens: c }) => {
                    prompt_tokens = p;
                    completion_tokens = c;
                    let chunk = StreamChunk::finish(&id, &model_name, prepared.created, reason);
                    yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                    guard.finished = true;
                    break;
                }
                Ok(StreamEvent::Failed { message }) => {
                    tracing::error!(request_id = %prepared.request_id, %message, "generation failed");
                    let chunk = StreamChunk::finish(&id, &model_name, prepared.created, FinishReason::Error);
                    yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                    guard.finished = true;
                    break;
                }
                Err(e) => {
                    tracing::error!(request_id = %prepared.request_id, error = %e, "stream error");
                    let chunk = StreamChunk::finish(&id, &model_name, prepared.created, FinishReason::Error);
                    yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                    break;
                }
            }
        }

        if include_usage {
            let usage = Usage::new(prompt_tokens, completion_tokens);
            let chunk = StreamChunk::usage_only(&id, &model_name, prepared.created, usage);
            yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
        }

        // `Sse` does not emit the OpenAI sentinel itself.
        yield Ok(Event::default().data("[DONE]"));
    }
}

/// Publishes a cancellation unless the request completed normally.
struct CancelOnDrop {
    transport: crate::nats::Transport,
    request_id: RequestId,
    finished: bool,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let transport = self.transport.clone();
        let id = self.request_id.clone();
        tracing::debug!(request_id = %id, "client disconnected; cancelling");
        tokio::spawn(async move {
            transport.cancel(&id).await;
        });
    }
}
