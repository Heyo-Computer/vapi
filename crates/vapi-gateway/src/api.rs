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
use vapi_proto::{DecisionRow, Job, JobKind, Subjects};

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
    /// When the gateway took the request, for the dashboard's timings.
    started: std::time::Instant,
    prompt_tokens: usize,
    /// Response-cache key, when the request is cacheable and the cache is on.
    cache_key: Option<String>,
}

impl Prepared {
    /// A cached answer for this request, if there is one.
    async fn cached(&self, st: &SharedState) -> Option<CachedResponse> {
        if !st
            .settings
            .response_cache
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return None;
        }
        let key = self.cache_key.as_deref()?;
        st.response_cache.as_ref()?.get(key).await
    }

    /// Record a finished answer for later requests, when it qualifies: the
    /// generation ran to a natural end, not a cancel or an error.
    fn remember(&self, st: &SharedState, tokens: Vec<String>, finish: FinishReason, usage: &Usage) {
        let (Some(key), Some(cache)) = (&self.cache_key, &st.response_cache) else {
            return;
        };
        if !st
            .settings
            .response_cache
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
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

/// How a request ended, for the dashboard's recent list.
struct Outcome<'a> {
    kind: &'static str,
    usage: &'a Usage,
    choices: usize,
    finish: FinishReason,
    /// Answered from the response cache without reaching a worker.
    cached: bool,
    streamed: bool,
}

fn record(st: &SharedState, prepared: &Prepared, o: Outcome<'_>) {
    st.stats.record(crate::state::RequestRecord {
        id: prepared.request_id.to_string(),
        kind: o.kind,
        prompt_tokens: o.usage.prompt_tokens,
        completion_tokens: o.usage.completion_tokens,
        choices: o.choices,
        duration_ms: prepared.started.elapsed().as_millis() as u64,
        finish: o.finish.as_openai().to_string(),
        cached: o.cached,
        streamed: o.streamed,
    });
}

fn prepare(
    st: &SharedState,
    kind: JobKind,
    prompt_tokens: Vec<u32>,
    params: vapi_core::SamplingParams,
    rows: Vec<DecisionRow>,
) -> ApiResult<Prepared> {
    let request_id = RequestId::new();
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // A decision job's prompt is many independent sequences end to end, so
    // the limit applies to the longest of them, not to their sum.
    let longest = if rows.is_empty() {
        prompt_tokens.len()
    } else {
        rows.iter().map(|r| r.len as usize).max().unwrap_or(0)
    };
    if longest >= st.cfg.model.max_context {
        return Err(Error::ContextLengthExceeded {
            tokens: longest,
            limit: st.cfg.model.max_context,
        }
        .into());
    }

    st.stats
        .started
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let n = prompt_tokens.len();
    // Decisions are pure functions of their input and would cache perfectly,
    // but the cache stores generated text; until it can hold an answer set,
    // saying so here beats storing something that can never be read back.
    let cache_key = (kind != JobKind::Decision
        && st.response_cache.is_some()
        && response_cache::cacheable(&params))
    .then(|| response_cache::key(&st.fingerprint, kind, &prompt_tokens, &params));
    Ok(Prepared {
        cache_key,
        started: std::time::Instant::now(),
        job: Job {
            request_id: request_id.clone(),
            model: st.model.clone(),
            kind,
            prompt_tokens,
            params,
            namespace: st.cfg.cache.default_namespace.clone(),
            reply_to: Subjects::stream(&request_id),
            enqueued_at_ms: created * 1000,
            rows,
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
    let mut params = req
        .to_sampling_params(st.cfg.model.max_context / 4)
        .map_err(|e| ApiError(Error::InvalidRequest(e)))?;
    // Compile the schema here so an unsupported one is a 400 on the
    // request rather than a failed generation later.
    if let Some(vapi_core::ResponseFormat::JsonSchema { schema }) = &params.response_format {
        vapi_engine::Schema::compile(schema)
            .map_err(|e| ApiError(Error::InvalidRequest(format!("response_format: {e}"))))?;
    }
    // A model that opens a reasoning block has to finish thinking before
    // its answer is forced into a shape.
    if params.response_format.is_some() {
        params.constraint_starts_after = st
            .output_format
            .as_ref()
            .and_then(|f| f.think_close)
            .map(str::to_string);
    }
    let tools = req
        .active_tools()
        .map_err(|e| ApiError(Error::InvalidRequest(e)))?;
    if tools.is_some() && st.output_format.is_none() {
        return Err(ApiError(Error::InvalidRequest(
            "this model has no tool-call format the gateway can parse".into(),
        )));
    }
    let tokens = st.tokenizer.encode_chat(&req.messages, tools)?;
    let prepared = prepare(&st, JobKind::Chat, tokens, params, Vec::new())?;

    let streaming = req.is_streaming();
    let include_usage = req.include_usage();
    let model_name = req.model.clone();
    let id = format!("chatcmpl-{}", prepared.request_id);
    let parser = st
        .output_format
        .clone()
        .map(|f| OutputParser::new(f, prepared.request_id.as_str()));

    if let Some(hit) = prepared.cached(&st).await {
        record(
            &st,
            &prepared,
            Outcome {
                kind: "chat",
                usage: &Usage::new(hit.prompt_tokens, hit.completion_tokens),
                choices: 1,
                finish: hit.finish,
                cached: true,
                streamed: streaming,
            },
        );
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
        let (out, reason, usage, logprobs, rest) = collect(st.clone(), prepared, parser).await?;
        if !rest.is_empty() {
            // Several completions: the extra choices are plain text, since
            // tool calls and reasoning are reported for the first only.
            let mut choices = vec![(out.content, reason)];
            choices.extend(rest);
            return Ok(Json(ChatCompletionResponse::with_choices(
                id, model_name, choices, usage,
            ))
            .into_response());
        }
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
    let prepared = prepare(&st, JobKind::Completion, tokens, params, Vec::new())?;
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
        let (out, reason, usage, logprobs, _) = collect(st.clone(), prepared, None).await?;
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

/// Point a chunk at the completion it belongs to.
fn set_choice_index(chunk: &mut StreamChunk, index: usize) {
    if let Some(c) = chunk.choices.first_mut() {
        c.index = index;
    }
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

/// The first choice in full, then the text and finish reason of each
/// further choice for `n > 1`.
type Collected = (
    ParsedOutput,
    FinishReason,
    Usage,
    Option<Vec<LogprobEntry>>,
    Vec<(String, FinishReason)>,
);

/// Subscribe, publish, and drain the whole stream into one response body.
async fn collect(
    st: SharedState,
    prepared: Prepared,
    parser: Option<OutputParser>,
) -> ApiResult<Collected> {
    let (mut stream, mut slot) = open_and_publish(&st, &prepared).await?;
    let choices = prepared.job.params.n.max(1);
    // One accumulator per choice; with `n == 1` this is the old path.
    let mut accs: Vec<Parsed> = (0..choices).map(|_| Parsed::new(parser.clone())).collect();
    let mut finishes: Vec<Option<FinishReason>> = vec![None; choices];
    let mut usage = Usage::new(prepared.prompt_tokens, 0);
    let mut logprobs: Option<Vec<LogprobEntry>> = None;
    let mut deltas: Vec<String> = Vec::new();
    let mut done = 0usize;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Started {
                cached_prefix_tokens,
            } => {
                slot.release();
                metrics::counter!("vapi_cached_prefix_tokens_total")
                    .increment(cached_prefix_tokens as u64);
            }
            // Only reachable when a decision worker is serving this model's
            // queue, which means the deployment is misconfigured rather than
            // that this request went wrong.
            StreamEvent::Decided { .. } => {
                return Err(Error::Engine(
                    "a decision worker answered a generation request;                      this model's queue has the wrong kind of worker on it"
                        .into(),
                )
                .into());
            }
            StreamEvent::Token {
                choice,
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
                if let Some(acc) = accs.get_mut(choice as usize) {
                    acc.push(&t);
                }
                if choice == 0 {
                    deltas.push(t);
                }
            }
            StreamEvent::Done {
                choice,
                reason,
                prompt_tokens,
                completion_tokens,
            } => {
                if let Some(f) = finishes.get_mut(choice as usize)
                    && f.is_none()
                {
                    *f = Some(reason);
                    done += 1;
                }
                // Usage accumulates over the choices; the prompt is shared
                // and counted once.
                usage = Usage::new(prompt_tokens, usage.completion_tokens + completion_tokens);
                if choice == 0 {
                    prepared.remember(&st, std::mem::take(&mut deltas), reason, &usage);
                }
                if done >= choices {
                    break;
                }
            }
            StreamEvent::Failed { message } => {
                st.stats
                    .failed
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(Error::Engine(message).into());
            }
        }
    }
    record(
        &st,
        &prepared,
        Outcome {
            kind: match prepared.job.kind {
                JobKind::Chat => "chat",
                JobKind::Completion => "completion",
                JobKind::Decision => "decision",
            },
            usage: &usage,
            choices,
            finish: finishes
                .first()
                .copied()
                .flatten()
                .unwrap_or(FinishReason::Stop),
            cached: false,
            streamed: false,
        },
    );
    let mut outs = Vec::with_capacity(choices);
    for (i, acc) in accs.into_iter().enumerate() {
        let reason = finishes
            .get(i)
            .copied()
            .flatten()
            .unwrap_or(FinishReason::Stop);
        outs.push(acc.finish(reason));
    }
    let (out, finish) = outs.remove(0);
    let rest = outs
        .into_iter()
        .map(|(o, f)| (o.content, f))
        .collect::<Vec<_>>();
    Ok((out, finish, usage, logprobs, rest))
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
        st.settings.first_token_timeout(),
        st.settings.stream_idle_timeout(),
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
        // One parser per choice: each completion has its own reasoning
        // block and its own tool calls.
        let mut accs: Vec<Parsed> = (0..prepared.job.params.n.max(1))
            .map(|_| Parsed::new(parser.clone()))
            .collect();
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
        let mut choices_left = prepared.job.params.n.max(1);

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
                Ok(StreamEvent::Decided { .. }) => {
                    // The deployment is misconfigured: a decision worker is
                    // consuming this model's generation queue.
                    tracing::error!(
                        request_id = %prepared.request_id,
                        "a decision worker answered a generation request"
                    );
                    let chunk = StreamChunk::finish(&id, &model_name, prepared.created, FinishReason::Error);
                    yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                    break;
                }
                Ok(StreamEvent::Token { choice, text, logprob, top_logprobs }) => {
                    completion_tokens += 1;
                    if choice == 0 && prepared.cache_key.is_some() {
                        deltas.push(text.clone());
                    }
                    let i = choice as usize;
                    let mut chunks = match accs.get_mut(i) {
                        Some(acc) => acc.stream(&id, &model_name, prepared.created, &text),
                        None => continue,
                    };
                    if let (Some(lp), Some(chunk)) = (logprob, chunks.first_mut()) {
                        *chunk = chunk.clone().with_logprobs(logprob_entry(
                            &st,
                            &text,
                            lp,
                            top_logprobs.as_deref().unwrap_or(&[]),
                        ));
                    }
                    for mut chunk in chunks {
                        set_choice_index(&mut chunk, i);
                        yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                    }
                }
                Ok(StreamEvent::Done { choice, reason, prompt_tokens: p, completion_tokens: c }) => {
                    let i = choice as usize;
                    if choice == 0 {
                        prompt_tokens = p;
                        completion_tokens = c;
                        prepared.remember(
                            &st,
                            std::mem::take(&mut deltas),
                            reason,
                            &Usage::new(p, c),
                        );
                    }
                    let (reason, tail) = match accs.get_mut(i) {
                        Some(acc) => acc.finish_stream(&id, &model_name, prepared.created, reason),
                        None => (reason, Vec::new()),
                    };
                    for mut chunk in tail {
                        set_choice_index(&mut chunk, i);
                        yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                    }
                    let chunk = StreamChunk::finish_for(&id, &model_name, prepared.created, i, reason);
                    yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
                    choices_left = choices_left.saturating_sub(1);
                    if choices_left > 0 {
                        continue;
                    }
                    record(
                        &st,
                        &prepared,
                        Outcome {
                            kind: "chat",
                            usage: &Usage::new(prompt_tokens, completion_tokens),
                            choices: prepared.job.params.n.max(1),
                            finish: reason,
                            cached: false,
                            streamed: true,
                        },
                    );
                    guard.finished = true;
                    break;
                }
                Ok(StreamEvent::Failed { message }) => {
                    tracing::error!(request_id = %prepared.request_id, %message, "generation failed");
                    st.stats.failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

// ---------------------------------------------------------------- decisions

/// `POST /v1/decisions` — every question answered in one forward pass.
///
/// No streaming and no sampling: the whole answer arrives at once, and the
/// same input always produces the same output. What the gateway does either
/// side of the worker is the part worth keeping here rather than there — the
/// sequence builder and the calibration are both things an operator refits
/// without touching the model.
pub async fn decisions(
    State(st): State<SharedState>,
    Json(req): Json<vapi_openai::DecisionRequest>,
) -> ApiResult<Response> {
    let format = st.decision.as_ref().ok_or_else(|| {
        ApiError(Error::InvalidRequest(
            "this deployment serves a generative model; /v1/decisions needs a decision checkpoint"
                .into(),
        ))
    })?;
    let bundle = st.tokenizer.bundle().ok_or_else(|| {
        ApiError(Error::InvalidRequest(
            "no tokenizer is loaded, so no question can be built".into(),
        ))
    })?;
    if req.questions.is_empty() {
        return Err(ApiError(Error::InvalidRequest(
            "no questions were asked".into(),
        )));
    }

    let state_text = req.state_text();
    // An English checkpoint does not degrade gracefully on a script it cannot
    // read: the published figure is 0.000 accuracy at 0.952 confidence on
    // Khmer, so no confidence threshold downstream can catch it. Reading the
    // script costs microseconds and is the only thing that can.
    let reading = vapi_core::script::read(&state_text);
    if !format.multilingual && !reading.readable_by_latin_model() {
        metrics::counter!("vapi_decision_unreadable_script_total").increment(1);
        tracing::warn!(
            script = %reading.script,
            share = reading.share,
            "this checkpoint cannot read this script; the answer will be confident and wrong. \
             Serve the multilingual checkpoint for it"
        );
    }
    let mut prompt_tokens = Vec::new();
    let mut rows = Vec::with_capacity(req.questions.len());
    let mut options_per_question = Vec::with_capacity(req.questions.len());
    for (id, question) in req.questions.iter() {
        let options = question
            .option_texts()
            .map_err(|e| ApiError(Error::InvalidRequest(format!("question {id:?}: {e}"))))?;
        let built = format
            .build(
                bundle,
                question.qtype,
                &question.instruction_text(),
                &options,
                &state_text,
            )
            .map_err(|e| ApiError(Error::InvalidRequest(format!("question {id:?}: {e}"))))?;
        if let Some(per) = built.option_tokens {
            // Answerable, but the caller should know the options were cut:
            // below about eight tokens each they stop being distinguishable
            // and accuracy falls off sharply.
            tracing::warn!(
                question = %id,
                options = options.len(),
                tokens_each = per,
                cramped = built.options_are_cramped(),
                "option texts were truncated to fit the question budget"
            );
        }
        rows.push(vapi_proto::DecisionRow {
            len: built.tokens.len() as u32,
            markers: built.markers,
            qtype: question.qtype,
        });
        prompt_tokens.extend(built.tokens);
        options_per_question.push(options.len());
    }

    let prepared = prepare(
        &st,
        JobKind::Decision,
        prompt_tokens,
        vapi_core::SamplingParams::default(),
        rows,
    )?;
    let id = format!("dec-{}", prepared.request_id);
    let model_name = req.model.clone().unwrap_or_else(|| st.cfg.model.id.clone());

    let (scores, prompt_tokens) = await_decision(&st, &prepared).await?;
    if scores.len() != req.questions.len() {
        return Err(ApiError(Error::Engine(format!(
            "worker answered {} of {} questions",
            scores.len(),
            req.questions.len()
        ))));
    }

    let calibration = &format.config.calibration;
    let answers: vapi_openai::Ordered<vapi_openai::Answer> = req
        .questions
        .iter()
        .zip(&scores)
        .zip(&options_per_question)
        .map(|(((qid, question), row), &options)| {
            (
                qid.clone(),
                shape_answer(
                    question,
                    row,
                    calibration.temperature(question.qtype, options),
                ),
            )
        })
        .collect();

    let usage = Usage::new(prompt_tokens, 0);
    record(
        &st,
        &prepared,
        Outcome {
            kind: "decision",
            usage: &usage,
            choices: answers.len(),
            finish: FinishReason::Stop,
            cached: false,
            streamed: false,
        },
    );
    st.stats
        .completed
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    Ok(Json(vapi_openai::DecisionResponse {
        id,
        object: "decision",
        created: prepared.created,
        model: model_name,
        answers,
        usage,
    })
    .into_response())
}

/// Publish and wait for the single terminal delta.
async fn await_decision(
    st: &SharedState,
    prepared: &Prepared,
) -> ApiResult<(Vec<vapi_proto::RowScores>, usize)> {
    let (mut stream, mut slot) = open_and_publish(st, prepared).await?;
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Started { .. } => slot.release(),
            StreamEvent::Decided {
                rows,
                prompt_tokens,
            } => return Ok((rows, prompt_tokens)),
            StreamEvent::Failed { message } => {
                st.stats
                    .failed
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(ApiError(Error::Engine(message)));
            }
            // A generating worker on a decision queue: the mirror of the
            // check on the other path, and the same misconfiguration.
            StreamEvent::Token { .. } | StreamEvent::Done { .. } => {
                return Err(ApiError(Error::Engine(
                    "a generative worker answered a decision request; \
                     this model's queue has the wrong kind of worker on it"
                        .into(),
                )));
            }
        }
    }
    Err(ApiError(Error::Engine(
        "the worker closed the stream without answering".into(),
    )))
}

/// Turn one question's raw logits into the answer its type calls for.
fn shape_answer(
    question: &vapi_openai::Question,
    row: &vapi_proto::RowScores,
    temperature: f32,
) -> vapi_openai::Answer {
    use vapi_openai::{Answer, round4};

    let p = vapi_core::calibrated_probabilities(&row.logits, temperature);
    let confidence = round4(vapi_core::confidence(&p));
    let act = round4(row.act as f64);
    let labels = question.labels();
    let probabilities: vapi_openai::Ordered<f64> = labels
        .iter()
        .cloned()
        .zip(p.iter().map(|&x| round4(x)))
        .collect();

    match question.qtype {
        vapi_core::QuestionType::Choice => {
            let best = p
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            Answer::Choice {
                choice: labels.get(best).cloned().unwrap_or_default(),
                probabilities,
                confidence,
                act_probability: act,
            }
        }
        vapi_core::QuestionType::Score => Answer::Score {
            // The expectation, not the argmax: an ordinal answer of 1.44 is
            // the useful one, and no single level carries it.
            score: round4(vapi_core::expected_level(&p)),
            legend: question.legend(),
            probabilities,
            confidence,
            act_probability: act,
        },
        // `false` is index 0 and `true` is index 1, fixed by the trained
        // option order, so the answer is p[1].
        vapi_core::QuestionType::Noul => Answer::Noul {
            noul: round4(p.get(1).copied().unwrap_or(0.0)),
            act_probability: act,
        },
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
