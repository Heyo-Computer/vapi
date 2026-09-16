# vapi

A Rust LLM inference server: OpenAI-compatible HTTP, a NATS JetStream work
queue, paged KV cache, and a prefix cache shared across **all** users.

The design target is vLLM's architecture — paged attention, continuous
batching, automatic prefix caching — with the request lifecycle organised
around a durable work queue so prompts are handled asynchronously and many
users are served concurrently.

## Status

The engine, cache, scheduler and full request path are built and tested. The
model itself is not yet wired up: the worker currently runs a **mock backend**,
so the system streams real tokens through the real pipeline but the tokens do
not come from real weights.

| Area | State |
|---|---|
| OpenAI API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) | done, streaming + non-streaming |
| NATS JetStream work queue, backpressure, ack heartbeats | done |
| Cancellation on client disconnect | done |
| Paged block pool, refcounting, LRU eviction | done, property-tested |
| Shared prefix cache (tier 1) | done — measured 63% hit rate across repeat prompts |
| Continuous batching, chunked prefill, preemption | done |
| Sampling (temperature/top-k/top-p/min-p/penalties/seeds) | done |
| Incremental detokenization | done, handles split multi-byte characters |
| **Real model execution (candle)** | **scaffolded, not implemented** |
| Response cache (tier 2), spill tier (tier 3) | not started |
| CUDA | scaffolded behind a feature flag |

150 tests, all passing, none requiring a GPU or a model download.

## Running it

```sh
just nats        # NATS with JetStream, via docker compose
just gw          # gateway on :8080
just worker      # engine worker, metrics on :9090
just smoke       # end-to-end streaming request
just metrics     # prefix-cache hit rate, KV utilization, queue depth
```

With no `model.path` in `vapi.toml`, both binaries fall back to a byte-level
tokenizer and the mock backend, so the whole path runs with nothing to
download.

```sh
curl -N localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

## Architecture

```
   client (openai sdk / curl)
        │  POST /v1/chat/completions  (SSE)
        ▼
   ┌─────────────────────┐   1. subscribe vapi.stream.<req_id>   (core NATS)
   │  vapi-gateway       │   2. publish  vapi.jobs.<model>       (JetStream)
   │  axum + tokenizer   │   3. relay deltas → SSE
   └─────────────────────┘
        │                    ▲ token deltas (core NATS, fire-and-forget)
        ▼ durable job        │
   ┌───────────────────────────────────────────────────┐
   │  vapi-worker (1..N)                               │
   │   pull consumer ──► Scheduler (continuous batch)  │
   │                      ├─ BlockPool  (paged KV)     │
   │                      ├─ PrefixCache (shared)      │
   │                      └─ ExecutionBackend          │
   └───────────────────────────────────────────────────┘
```

### Durable for work, ephemeral for tokens

Prompts go through JetStream because losing one loses a user's request. Token
deltas go over plain core NATS, because a JetStream publish is an acknowledged,
persisted write — doing that per token at a few-milliseconds cadence would
fsync thousands of times a second for data whose entire lifetime is one HTTP
connection. If the SSE connection dies the deltas are worthless anyway.

Because core NATS never replays, two things follow, and both are implemented:
the gateway **subscribes before publishing** the job (a delta published before
the subscription exists is simply gone, and on a cache hit the first token can
arrive in single-digit milliseconds), and every delta carries a **sequence
number** so a dropped message surfaces as an error rather than a response with
a word silently missing.

### Backpressure

The pull consumer's `max_ack_pending` is the engine's concurrency limit, so
NATS stops delivering when a worker is full and bursts queue in the stream
rather than in the worker's memory. Generations routinely outlive `ack_wait`,
so the worker sends `AckKind::Progress` heartbeats and acks only at
`finish_reason` — without that, JetStream assumes the worker died and hands the
prompt to another one, duplicating the user's answer.

### The shared prefix cache

Blocks are content-addressed by a **chained** hash: each block's hash mixes in
its parent's, so two sequences share a block only when every token before it
also matches. The model id, weights fingerprint, dtype, any adapter, and a
tenant namespace all fold into the chain — omitting any of them serves one
model's KV as another's, which is silent corruption rather than a miss.

> **Cross-tenant note.** A globally shared cache is a timing side channel:
> time-to-first-token reveals whether *someone* recently submitted a given
> prefix. vLLM has the same property. `cache.default_namespace` is a single
> shared namespace for maximum reuse; a tenant needing isolation gets its own.

## Continuing on a GPU box

The engine is complete and tested against `MockBackend`, so the remaining work
is implementing `ExecutionBackend` for real weights. Start in
`crates/vapi-backend-candle/`.

1. **Prototype the KV scatter first** (`src/cache.rs::write_kv_to_cache`). It
   is the one op with no confirmed off-the-shelf answer in candle and it sits
   in the innermost loop, so pin down `scatter`'s index-rank semantics on CPU
   before building on it.
2. **Vendor the model.** Copy `candle-transformers-0.11.0/src/models/llama.rs`
   (534 lines) and keep `Config`, RMSNorm, the MLP, the RoPE tables, weight
   loading and `repeat_kv`. Replace exactly two things: `Cache` becomes the
   paged cache, and `CausalSelfAttention::forward(x, index_pos, cache)` becomes
   `forward(x, &ForwardBatch, &mut PagedKvCache)` — the stock signature takes a
   single `index_pos` and so structurally cannot batch sequences sitting at
   different positions. About 150 of those 534 lines change.
3. **Golden-token test before anything else.** A fixed prompt, greedy, must
   reproduce a token-id sequence recorded from HF `transformers`. Model bugs
   here produce fluent, subtly wrong text, not crashes.
4. **Then CUDA.** `--features cuda` wires up
   `flash_attn_varlen_paged_windowed`, which handles prefill and decode in one
   call. The engine already emits the exact layout it needs.

Two constraints are already baked in so they are not surprises later:
`BLOCK_SIZE` is **32** (the kernel rejects a `page_block_size` that is not a
multiple of 32 — vLLM's 16 would fail), and the KV buffers are laid out
`(num_blocks, block_size, num_kv_heads, head_dim)`, which is exactly what the
kernel reads.

Note `candle-flash-attn` compiles FlashAttention from CUDA source: budget tens
of minutes and plenty of RAM, and do it once early rather than discovering it
mid-milestone.

## Testing

`MockBackend` is the load-bearing piece: it makes the scheduler, cache, NATS
protocol and HTTP surface testable in milliseconds with no GPU and no weights.
It also **validates every batch it receives** — duplicate cache slots,
block-table coverage, position contiguity, cumulative-length consistency — so a
scheduler test that never looks at a tensor still catches a malformed block
table.

The highest-value tests:

- `prefix_cache_hits_do_not_change_the_output` — a warm cache must be a pure
  performance win. If this fails, `slot_mapping` or `num_computed` is off by
  one and the server produces fluent, wrong text.
- `pool_invariants_hold_under_arbitrary_traces` — a proptest that caught a real
  index-corruption bug on its first run.
- `running_out_of_blocks_preempts_rather_than_failing` — oversubscribed pool,
  every request still completes, no blocks leaked.

## Layout

```
crates/
  vapi-core/            ids, sampling params, config, errors
  vapi-openai/          OpenAI wire types
  vapi-proto/           NATS subjects, job/delta envelopes
  vapi-tokenize/        tokenizer + chat templates (minijinja)
  vapi-cache/           block pool, chained hashing, prefix cache
  vapi-engine/          scheduler, sampler, detokenizer, backend trait
  vapi-backend-candle/  real model execution (scaffold)
  vapi-gateway/         axum HTTP server
  vapi-worker/          engine host
```
