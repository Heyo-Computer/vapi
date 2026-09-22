# vapi

A Rust LLM inference server: OpenAI-compatible HTTP, a NATS JetStream work
queue, paged KV cache, and a prefix cache shared across **all** users.

The design target is vLLM's architecture — paged attention, continuous
batching, automatic prefix caching — with the request lifecycle organised
around a durable work queue so prompts are handled asynchronously and many
users are served concurrently.

## Status

The engine, cache, scheduler and full request path are built and tested, and
real Llama-architecture weights run through candle on the CPU (f32, greedy
output matching Hugging Face `transformers` token for token) and on CUDA
(bf16, through `candle-flash-attn`'s paged varlen kernel). Without a
`model.path` the worker runs a **mock backend**, so the whole pipeline is
still exercisable with nothing downloaded.

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
| Stop strings | done, held back across token boundaries |
| Real model execution (candle, CPU) | done — paged Llama matches HF goldens exactly in f32 |
| LFM2.5-2.6B (Liquid AI): gated short-conv layers + GQA attention in one paged cache | done — the 2.6B model matches HF `transformers` token for token in f32 on the CPU, matches or ties in bf16 on the GPU, and serves through the gateway |
| Laguna architecture (poolside): MoE, gated attention, sliding window, YaRN | done at tiny scale — matches poolside's reference within 1e-4; the real 33B model needs a ≥48 GB GPU |
| Response cache (tier 2) | done — exact-match cache of deterministic completions in NATS KV, hits never touch a worker |
| Tool calling + reasoning (LFM2.5 format) | done — `tools` rendered through the chat template, calls parsed back into OpenAI `tool_calls`, thinking into `reasoning_content` |
| Spill tier (tier 3) | done — evicted blocks to host memory then disk, restored on the next request; off by default, close to break-even on this GPU (`benchmark/README.md`) |
| Multi-worker cache affinity | done — partitioned job subjects routed by the conversation's first block, plus a worker registry; hit rate 0.40 → 0.49 on two workers |
| Production behaviour: failed-step policy, 503 + Retry-After on overload, graceful drain, logprobs | done — each checked end to end against the mock and, for cancel and preemption, against the real model |
| CUDA (FlashAttention paged kernel, bf16) | done — kernel checked against the CPU reference; goldens match or diverge only at exact ties |
| Benchmark against vLLM | done — `benchmark/README.md`; parity to batch 8, vLLM 1.2x ahead at batch 64 (was 6.6x) |

152 tests pass with no features, 34 with `--features candle`, and 37 more
with `--features cuda` on a GPU. Six further golden tests run when a model
is present (see [Testing](#testing)), including two that check the
tool-call prompt against Hugging Face.

## Running it

`docs/running-a-local-model.md` is the full walkthrough: getting weights,
sizing the KV cache, CPU and GPU builds, talking to it with the OpenAI SDK,
and a troubleshooting table. `docs/PLAN.md` is the roadmap.

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

To serve a real model, download a Llama-architecture checkpoint and point
both binaries at it. The gateway uses the directory for the tokenizer and
chat template; the worker also loads the weights, so the worker must be built
with `--features candle` (CPU) or `--features cuda` (GPU):

```sh
hf download HuggingFaceTB/SmolLM2-135M-Instruct --local-dir ~/models/smollm2-135m-instruct
# in vapi.toml:  [model] path = "/home/you/models/smollm2-135m-instruct"  num_blocks = 128
cargo run --release -p vapi-gateway
cargo run --release -p vapi-worker --features candle
```

`model.num_blocks` sizes the KV cache and defaults to 2048 blocks (32 tokens
each). On CUDA, `CandleBackend` can size it from free device memory ×
`model.kv_cache_fraction` instead, but only when `num_blocks` is unset, and
the config default is not unset; that path is wired but not reachable from
`vapi.toml` yet.

Building with CUDA compiles FlashAttention from source once (tens of minutes;
cached in `CANDLE_FLASH_ATTN_BUILD_DIR`, which must already exist):

```sh
mkdir -p ~/.cache/candle-flash-attn
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_COMPUTE_CAP=120                       # 80 A100 · 86 RTX 30 · 89 RTX 40 · 90 H100 · 120 RTX 50
export CANDLE_FLASH_ATTN_BUILD_DIR=$HOME/.cache/candle-flash-attn
cargo run --release -p vapi-worker --features cuda
```

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

## CUDA

`--features cuda` turns on `candle-core/cuda`, `candle-nn/cuda` and
`candle-transformers/cuda` (the latter two matter: without them RMSNorm,
softmax and RoPE have no device kernels and fail at runtime with "no cuda
implementation") and pulls in `candle-flash-attn`. The attention call is
`flash_attn_varlen_paged_windowed` with `window_size_right = Some(0)`, which
is what makes it causal; `None` on both sides is bidirectional, and
`a_bidirectional_window_is_not_causal` pins that.

Verified on an RTX 5060 Ti (sm_120, CUDA 12.8, driver 595): the kernel
matches `paged_attention_cpu` within 1e-2 in bf16 and 1e-3 in f16 on a mixed
batch (prefill chunk over a cached prefix plus a decode, non-contiguous
blocks), and the SmolLM2 goldens in bf16 either match HF exactly or diverge
at a logit gap of 0.0000, which is a tie rather than a bug.

Two constraints are baked in so they are not surprises: `BLOCK_SIZE` is
**32** (the kernel rejects a `page_block_size` that is not a multiple of 32 —
vLLM's 16 would fail), and the KV buffers are laid out
`(num_blocks, block_size, num_kv_heads, head_dim)`, which is exactly what the
kernel reads.

Measured against vLLM on the same card and client with LFM2.5-2.6B in bf16,
greedy (`benchmark/README.md` has the setup, both models, and the
step-by-step phase 4 history):

| Concurrent | TTFT p50 vapi / vLLM | Inter-token p50 vapi / vLLM | Throughput vapi / vLLM |
|---|---|---|---|
| 1 | 16 / 24 ms | 13.9 / 13.5 ms | 71 / 74 tok/s (1.0x) |
| 8 | 48 / 316 ms | 14.4 / 13.9 ms | 540 / 490 tok/s (0.9x) |
| 32 | 109 / 138 ms | 15.8 / 14.4 ms | 1,888 / 2,076 tok/s (1.1x) |
| 64 | 222 / 146 ms | 17.8 / 15.3 ms | 3,240 / 3,761 tok/s (1.2x) |

Parity up to batch 8; vLLM is 1.2x ahead at batch 64, with CUDA graphs on
(`model.cuda_graphs = true`) and the fused FFN and conv kernels. Sampling
with the model's recommended settings (temperature 0.1, top-k 50,
repetition penalty 1.1) runs at the same speed as greedy: candidates are
selected on the device and only they cross to the host. Sampled requests (the model card's
temperature 0.1, top-k 50, repetition penalty 1.1) run at 74 ms per token
at batch 64 after a sampler rewrite that took them from 335 ms; they still
copy full logits to the host.

One operational note: the JetStream consumer is durable, and the worker now
uses `create_consumer` so a changed `worker.max_concurrent_seqs` updates the
consumer's `max_ack_pending`. With get-or-create the first run's value stuck
and silently capped concurrency.

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
- `models::llama::tests::*` (`--features candle`) — the paged Llama's logits
  match the unmodified candle-transformers Llama on the same random weights,
  through chunked prefill, decode, and a mixed batch.
- `attention::cuda_tests::*` (`--features cuda`) — the FlashAttention paged
  kernel against the CPU reference on a mixed batch, with and without a
  sliding window, and proof that the bidirectional window differs.
- `models::lfm2::tests::*` (`--features candle`) — the paged LFM2 against
  transformers on a random fixture: whole prompt, decode, every chunk split
  across the conv window, mixed batch.
- `models::laguna::tests::*` and `tests/tiny_backends.rs` (`--features
  candle`) — the paged Laguna against poolside's `modeling_laguna.py` on
  `Laguna-tiny-per-element` and on a random clone of Laguna-XS-2.1's
  feature set, within 1e-4, plus greedy ids through the scheduler. Fixtures
  come from `tools/gen_laguna_goldens.py`; the tests skip without them.

### Golden tests against a real model

`tests/goldens/*.json` hold, per fixture, the chat template as Hugging Face
renders it, the prompt ids, and the first 64 greedy tokens from
`transformers` in f32. They were generated with `tools/gen_goldens.py` from
`HuggingFaceTB/SmolLM2-135M-Instruct` (Llama architecture, 135M parameters,
ungated). `crates/vapi-backend-candle/tests/real_model.rs` checks all three
exactly on the CPU, plus concurrent requests through the scheduler and
prefix-cache-on versus off. On a CUDA device (`--features cuda`) it runs in
bf16 and switches to tolerance mode: it prints the first divergent token and
the top-1/top-2 logit gap there, failing only when the gap exceeds 0.5. The tests skip unless a model is at
`~/models/smollm2-135m-instruct` or `VAPI_TEST_MODEL_DIR`:

```sh
cargo test -p vapi-backend-candle --features candle --test real_model -- --nocapture
```

To regenerate the goldens (needs `torch` and `transformers`; `uv` handles it):

```sh
uv run --index https://download.pytorch.org/whl/cpu --with torch --with transformers --with safetensors \
    tools/gen_goldens.py ~/models/smollm2-135m-instruct --model-id HuggingFaceTB/SmolLM2-135M-Instruct
```

## Layout

```
crates/
  vapi-core/            ids, sampling params, config, errors
  vapi-openai/          OpenAI wire types
  vapi-proto/           NATS subjects, job/delta envelopes
  vapi-tokenize/        tokenizer + chat templates (minijinja)
  vapi-cache/           block pool, chained hashing, prefix cache
  vapi-engine/          scheduler, sampler, detokenizer, backend trait
  vapi-backend-candle/  paged Llama on candle, CPU reference + CUDA attention
tests/goldens/          HF transformers outputs the real-model tests compare against
tools/gen_goldens.py    regenerates them
benchmark/              bench.py, results/, and the vapi-vs-vLLM comparison
  vapi-gateway/         axum HTTP server
  vapi-worker/          engine host
```
