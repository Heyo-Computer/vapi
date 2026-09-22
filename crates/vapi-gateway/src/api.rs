use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response, Sse, sse::Event};
use futures::stream::Stream;
use vapi_core::{Error, FinishReason, RequestId};
use vapi_openai::{
    ChatCompletionRequest, ChatCompletionResponse, CompletionRequest, CompletionResponse,
    LogprobEntry, Model, ModelList, StreamChunk, ToolCall, Usage,
};
use vapi_proto::{Job, JobKind, Subjects};

use crate::output::{OutputParser, Piece};
use crate::response_cache::{self, CachedResponse};
use crate::state::{QueueSlot, SharedState};
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
        let mut resp = (status, Json(body)).into_response();
        if matches!(self.0, Error::Overloaded) {
            // What a client's backoff keys on. One second: the queue moves
            // at a step every few tens of milliseconds, so by then a slot
            // has very likely opened.
            resp.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                "1".parse().expect("static header"),
            );
        }
        resp
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
    /// Response-cache key, when the request is cacheable and the cache is on.
    cache_key: Option<String>,
}

impl Prepared {
    /// A cached answer for this request, if there is one.
    async fn cached(&self, st: &SharedState) -> Option<CachedResponse> {
        let key = self.cache_key.as_deref()?;
        st.response_cache.as_ref()?.get(key).await
    }

    /// Record a finished answer for later requests, when it qualifies: the
    /// generation ran to a natural end, not a cancel or an error.
    fn remember(&self, st: &SharedState, tokens: Vec<String>, finish: FinishReason, usage: &Usage) {
        let (Some(key), Some(cache)) = (&self.cache_key, &st.response_cache) else {
            return;
        };
        if !matches!(finish, FinishReason::Stop | FinishReason::Length) {
            return;
        }
        cache.put(
            key.clone(),
            CachedResponse {
                tokens,
                finish,
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
            },
        );
    }
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
    let cache_key = (st.response_cache.is_some() && response_cache::cacheable(&params))
        .then(|| response_cache::key(&st.fingerprint, kind, &prompt_tokens, &params));
    Ok(Prepared {
        cache_key,
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

/// Turn a worker's per-token logprobs into an OpenAI entry. The sampled
/// token's text is what was streamed; alternatives are decoded on their own.
fn logprob_entry(
    st: &SharedState,
    text: &str,
    logprob: f32,
    top: &[vapi_proto::TokenLogprob],
) -> LogprobEntry {
    let top = top
        .iter()
        .map(|t| {
            let s = st.tokenizer.decode(&[t.token_id]);
            (s, t.logprob)
        })
        .collect();
    LogprobEntry::new(text.to_string(), logprob, top)
}

pub async fn chat_completions(
    State(st): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> ApiResult<Response> {
    let params = req
        .to_sampling_params(st.cfg.model.max_context / 4)
        .map_err(|e| ApiError(Error::InvalidRequest(e)))?;
    let tools = req
        .active_tools()
        .map_err(|e| ApiError(Error::InvalidRequest(e)))?;
    if tools.is_some() && st.output_format.is_none() {
        return Err(ApiError(Error::InvalidRequest(
            "this model has no tool-call format the gateway can parse".into(),
        )));
    }
    let tokens = st.tokenizer.encode_chat(&req.messages, tools)?;
    let prepared = prepare(&st, JobKind::Chat, tokens, params)?;

    let streaming = req.is_streaming();
    let include_usage = req.include_usage();
    let model_name = req.model.clone();
    let id = format!("chatcmpl-{}", prepared.request_id);
    let parser = st
        .output_format
        .clone()
        .map(|f| OutputParser::new(f, prepared.request_id.as_str()));

    if let Some(hit) = prepared.cached(&st).await {
        return Ok(replay_chat(
            hit,
            prepared,
            parser,
            id,
            model_name,
            streaming,
            include_usage,
        ));
    }
    if streaming {
        Ok(sse_response(st, prepared, parser, id, model_name, include_usage).await?)
    } else {
        let (out, reason, usage, logprobs) = collect(st, prepared, parser).await?;
        let mut resp = ChatCompletionResponse::new(id, model_name, out.content, reason, usage)
            .with_parsed(out.reasoning, out.tool_calls);
        if let Some(lp) = logprobs {
            resp = resp.with_logprobs(lp);
        }
        Ok(Json(resp).into_response())
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

    if let Some(hit) = prepared.cached(&st).await {
        if req.is_streaming() {
            return Ok(replay_chat(
                hit, prepared, None, id, model_name, true, false,
            ));
        }
        let usage = Usage::new(hit.prompt_tokens, hit.completion_tokens);
        return Ok(Json(CompletionResponse::new(
            id,
            model_name,
            hit.text(),
            hit.finish,
            usage,
        ))
        .into_response());
    }
    if req.is_streaming() {
        Ok(sse_response(st, prepared, None, id, model_name, false).await?)
    } else {
        let (out, reason, usage, logprobs) = collect(st, prepared, None).await?;
        let mut resp = CompletionResponse::new(id, model_name, out.content, reason, usage);
        if let Some(lp) = logprobs {
            resp = resp.with_logprobs(&lp);
        }
        Ok(Json(resp).into_response())
    }
}

/// Answer from the response cache: the whole body at once, or the cached
/// deltas replayed as SSE chunks with the same shape a live stream has.
fn replay_chat(
    hit: CachedResponse,
    prepared: Prepared,
    parser: Option<OutputParser>,
    id: String,
    model_name: String,
    streaming: bool,
    include_usage: bool,
) -> Response {
    let usage = Usage::new(hit.prompt_tokens, hit.completion_tokens);
    if !streaming {
        let mut acc = Parsed::new(parser);
        for t in &hit.tokens {
            acc.push(t);
        }
        let (out, finish) = acc.finish(hit.finish);
        return Json(
            ChatCompletionResponse::new(id, model_name, out.content, finish, usage)
                .with_parsed(out.reasoning, out.tool_calls),
        )
        .into_response();
    }
    let created = prepared.created;
    let body = async_stream::stream! {
        let data = |c: &StreamChunk| Ok::<_, std::convert::Infallible>(
            Event::default().data(serde_json::to_string(c).unwrap_or_default()),
        );
        let mut acc = Parsed::new(parser);
        yield data(&StreamChunk::role(&id, &model_name, created));
        for t in hit.tokens {
            for chunk in acc.stream(&id, &model_name, created, &t) {
                yield data(&chunk);
            }
        }
        let (finish, tail) = acc.finish_stream(&id, &model_name, created, hit.finish);
        for chunk in tail {
            yield data(&chunk);
        }
        yield data(&StreamChunk::finish(&id, &model_name, created, finish));
        if include_usage {
            yield data(&StreamChunk::usage_only(&id, &model_name, created, usage));
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    Sse::new(body).into_response()
}

/// The parsed answer of one request.
#[derive(Default)]
struct ParsedOutput {
    content: String,
    reasoning: Option<String>,
    tool_calls: Vec<ToolCall>,
}

/// Runs deltas through the model's output parser when there is one, and
/// through nothing when there is not.
struct Parsed {
    parser: Option<OutputParser>,
    out: ParsedOutput,
}

impl Parsed {
    fn new(parser: Option<OutputParser>) -> Self {
        Self {
            parser,
            out: ParsedOutput::default(),
        }
    }

    fn take(&mut self, pieces: Vec<Piece>) {
        for p in pieces {
            match p {
                Piece::Reasoning(r) => self
                    .out
                    .reasoning
                    .get_or_insert_with(String::new)
                    .push_str(&r),
                Piece::Content(c) => self.out.content.push_str(&c),
                Piece::ToolCalls(calls) => self.out.tool_calls.extend(calls),
            }
        }
    }

    /// Accumulate one delta (non-streaming).
    fn push(&mut self, text: &str) {
        let pieces = match &mut self.parser {
            Some(p) => p.push(text),
            None => {
                self.out.content.push_str(text);
                return;
            }
        };
        self.take(pieces);
    }

    fn finish(mut self, reason: FinishReason) -> (ParsedOutput, FinishReason) {
        let Some(mut p) = self.parser.take() else {
            return (self.out, reason);
        };
        let pieces = p.finish();
        self.take(pieces);
        let finish = if p.saw_tool_calls() {
            FinishReason::ToolCalls
        } else {
            reason
        };
        (self.out, finish)
    }

    fn chunks(id: &str, model: &str, created: u64, pieces: Vec<Piece>) -> Vec<StreamChunk> {
        pieces
            .into_iter()
            .map(|piece| match piece {
                Piece::Reasoning(r) => StreamChunk::reasoning(id, model, created, r),
                Piece::Content(c) => StreamChunk::content(id, model, created, c),
                Piece::ToolCalls(calls) => StreamChunk::tool_calls(id, model, created, &calls),
            })
            .collect()
    }

    /// One delta as the chunks to stream for it.
    fn stream(&mut self, id: &str, model: &str, created: u64, text: &str) -> Vec<StreamChunk> {
        match &mut self.parser {
            Some(p) => Self::chunks(id, model, created, p.push(text)),
            None => vec![StreamChunk::content(id, model, created, text)],
        }
    }

    fn finish_stream(
        &mut self,
        id: &str,
        model: &str,
        created: u64,
        reason: FinishReason,
    ) -> (FinishReason, Vec<StreamChunk>) {
        let Some(p) = &mut self.parser else {
            return (reason, Vec::new());
        };
        let tail = Self::chunks(id, model, created, p.finish());
        let finish = if p.saw_tool_calls() {
            FinishReason::ToolCalls
        } else {
            reason
        };
        (finish, tail)
    }
}

type Collected = (ParsedOutput, FinishReason, Usage, Option<Vec<LogprobEntry>>);

/// Subscribe, publish, and drain the whole stream into one response body.
async fn collect(
    st: SharedState,
    prepared: Prepared,
    parser: Option<OutputParser>,
) -> ApiResult<Collected> {
    let (mut stream, mut slot) = open_and_publish(&st, &prepared).await?;
    let mut acc = Parsed::new(parser);
    let mut finish = FinishReason::Stop;
    let mut usage = Usage::new(prepared.prompt_tokens, 0);
    let mut logprobs: Option<Vec<LogprobEntry>> = None;
    let mut deltas: Vec<String> = Vec::new();

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Started {
                cached_prefix_tokens,
            } => {
                slot.release();
                metrics::counter!("vapi_cached_prefix_tokens_total")
                    .increment(cached_prefix_tokens as u64);
            }
            StreamEvent::Token {
                text: t,
                logprob,
                top_logprobs,
            } => {
                if let Some(lp) = logprob {
                    logprobs.get_or_insert_with(Vec::new).push(logprob_entry(
                        &st,
                        &t,
                        lp,
                        top_logprobs.as_deref().unwrap_or(&[]),
                    ));
                }
                acc.push(&t);
                deltas.push(t);
            }
            StreamEvent::Done {
                reason,
                prompt_tokens,
                completion_tokens,
            } => {
                finish = reason;
                usage = Usage::new(prompt_tokens, completion_tokens);
                prepared.remember(&st, std::mem::take(&mut deltas), finish, &usage);
                break;
            }
            StreamEvent::Failed { message } => return Err(Error::Engine(message).into()),
        }
    }
    let (out, finish) = acc.finish(finish);
    Ok((out, finish, usage, logprobs))
}

/// Open the token stream first, then publish the job. The queue slot is
/// taken before anything is published, so an overloaded gateway refuses
/// without touching NATS.
async fn open_and_publish(
    st: &SharedState,
    prepared: &Prepared,
) -> ApiResult<(TokenStream, QueueSlot)> {
    let slot = st.queue_slot()?;
    let stream = TokenStream::open(
        &st.transport.client,
        &prepared.request_id,
        Duration::from_secs(st.cfg.gateway.first_token_timeout_secs),
        Duration::from_secs(st.cfg.gateway.stream_idle_timeout_secs),
    )
    .await?;
    st.transport
        .publish_job(&st.model, &prepared.job, st.cfg.nats.job_partitions)
        .await?;
    Ok((stream, slot))
}

async fn sse_response(
    st: SharedState,
    prepared: Prepared,
    parser: Option<OutputParser>,
    id: String,
    model_name: String,
    include_usage: bool,
) -> ApiResult<Response> {
    let (stream, slot) = open_and_publish(&st, &prepared).await?;
    let body = sse_stream(
        st,
        stream,
        slot,
        prepared,
        parser,
        id,
        model_name,
        include_usage,
    );
    Ok(Sse::new(body).into_response())
}

#[allow(clippy::too_many_arguments)]
fn sse_stream(
    st: SharedState,
    mut stream: TokenStream,
    mut slot: QueueSlot,
    prepared: Prepared,
    parser: Option<OutputParser>,
    id: String,
    model_name: String,
    include_usage: bool,
) -> impl Stream<Item = std::result::Result<Event, std::convert::Infallible>> {
    async_stream::stream! {
        let mut acc = Parsed::new(parser);
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
        let mut deltas: Vec<String> = Vec::new();

        while let Some(event) = stream.next().await {
            match event {
                Ok(StreamEvent::Started { cached_prefix_tokens }) => {
                    slot.release();
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
                Ok(StreamEvent::Token { text, logprob, top_logprobs }) => {
                    completion_tokens += 1;
                    if prepared.cache_key.is_some() {
                        deltas.push(text.clone());
                    }
                    let mut chunks = acc.stream(&id, &model_name, prepared.created, &text);
                    if let (Some(lp), Some(chunk)) = (logprob, chunks.first_mut()) {
                        *chunk = chunk.clone().with_logprobs(logprob_entry(
                            &st,
                            &text,
                            lp,
                            top_logprobs.as_deref().unwrap_or(&[]),
                        ));
                    }
                    for chunk in chunks {
                        yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                    }
                }
                Ok(StreamEvent::Done { reason, prompt_tokens: p, completion_tokens: c }) => {
                    prompt_tokens = p;
                    completion_tokens = c;
                    prepared.remember(
                        &st,
                        std::mem::take(&mut deltas),
                        reason,
                        &Usage::new(p, c),
                    );
                    let (reason, tail) = acc.finish_stream(&id, &model_name, prepared.created, reason);
                    for chunk in tail {
                        yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overload_is_a_503_with_retry_after() {
        let resp = ApiError(Error::Overloaded).into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok()),
            Some("1")
        );
        let resp = ApiError(Error::Engine("x".into())).into_response();
        assert!(
            resp.headers()
                .get(axum::http::header::RETRY_AFTER)
                .is_none()
        );
    }
}
