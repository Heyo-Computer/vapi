# vapi improvement plan: serving LFM2.5-2.6B

Written 2026-09-21, replacing the Laguna plan. The destination is
`LiquidAI/LFM2.5-2.6B`: 2.6B parameters, bf16 weights of 5.4 GB, ungated,
supported natively by transformers and vLLM. It fits this machine's RTX
5060 Ti (16 GB) with room for a large KV cache, and in f32 on the CPU
(10.8 GB of the 31 GB RAM), so every phase below can be finished and
measured here. Every phase has a "done when" you can check, and every
phase adds tests in the style of the existing ones.

(The Laguna work stays in the tree: `models/laguna.rs`, its fixtures and
tests are a proven port of that architecture, but the 33B model needs a
≥48 GB GPU and its FP8 loader was not started.)

## What LFM2.5 needs that vapi does not have

LFM2 is a hybrid: 22 of its 30 layers are short gated convolutions, 8 are
grouped-query attention with per-head q/k norms. Per layer: `operator_norm`
→ operator (attention or conv) → residual → `ffn_norm` → SwiGLU MLP
(`w1`, `w3`, `w2`) → residual; then `embedding_norm` and a tied `lm_head`.

1. **The short conv operator.** `in_proj` maps hidden → 3×hidden, split
   into `B`, `C`, `x`; `h = B * x`; a causal depthwise conv over time with
   kernel `conv_L_cache = 3`; `y = C * conv(h)`; `out_proj`. Its state is
   the last two `h` rows per sequence, which is not a KV cache. The paged
   design absorbs it cleanly: store `h` per token at its cache slot in a
   per-layer buffer of width `hidden`, and compute the conv for a chunk by
   gathering the two preceding rows through the block table. That keeps
   chunked prefill, prefix caching and preemption exact with no new state
   type; the cost is 4 KB per token per conv layer, more than a KV pair.
2. **Per-layer cache layouts.** `PagedKvCache` allocates one shape for
   every layer; conv layers need `(num_blocks, 32, 1, hidden)` for `h` and
   no V. Sizing has to count both kinds.
3. **Attention details**: `q_layernorm`/`k_layernorm` per head, RoPE with
   theta 10,000,000, projections named `out_proj`, head_dim 64, 32 heads
   over 8 KV heads.
4. **Serving details**: the chat template lives in `chat_template.jinja`,
   not `tokenizer_config.json`; BOS `<|startoftext|>` is emitted by the
   template; EOS is `<|im_end|>` (124900); the template uses macros,
   `tojson`, `is mapping` / `is iterable` tests and a `preserve_thinking`
   variable; generation defaults are temperature 0.1, top-k 50, repetition
   penalty 1.1. 131K context.

## Phase 1 — architecture, proven at tiny scale

Goal: `PagedLfm2::forward(&ForwardBatch, &mut PagedKvCache)` in
`crates/vapi-backend-candle/src/models/lfm2.rs` matching transformers'
`Lfm2ForCausalLM` under 1e-4 in f32.

1. Goldens: `tools/gen_lfm2_goldens.py` builds a random tiny config
   (conv and attention layers, GQA, kernel 3) and records logits for a
   whole prompt, a decode step and 16 greedy ids. Done when the fixture
   and `tests/goldens/lfm2-tiny.json` exist.
2. Port the model with the paged conv above. Done when the fixture
   matches for a whole prompt, a decode step, chunked prefill (a chunk
   boundary inside the conv window is the case that matters) and a mixed
   batch.
3. `PagedKvCache::with_layouts`, `write_k_to_cache`, `CandleBackend`
   dispatch on `model_type = "lfm2"`, cache sizing that counts conv
   buffers. Done when `tests/tiny_backends.rs`-style greedy ids match
   through the scheduler.
4. CUDA: the attention layers use the flash kernel; the conv is plain
   tensor ops on either device. Done when the fixture matches on the GPU
   in bf16 within bf16 tolerance.

**Status 2026-09-21: phase 1 done.** `models/lfm2.rs` matches
transformers within 1e-4 on the fixture for a whole prompt, a decode step,
every chunk split across the conv window, and a mixed batch; the backend
dispatches on `model_type` and reproduces `generate()` ids through the
scheduler; the fixture runs on the GPU in bf16 within tolerance. Cache
layouts are per layer (`LayerLayout::Kv` / `Rows`) and sizing counts both.

## Phase 2 — the real model

1. `TokenizerBundle` reads `chat_template.jinja`; the template renders
   through minijinja identically to `apply_chat_template`. Done when
   `tools/gen_goldens.py` fixtures for LFM2.5-2.6B pass the template and
   prompt-id checks exactly. **Done**: it needed two things minijinja
   lacks, transformers' `{% generation %}` training-mask tag (stripped) and
   Python dict methods like `message.get` (minijinja-contrib's pycompat
   hook). The rendered prompt and the 22–26 prompt ids match HF exactly.
   Also learned: HF `generate()` applies `generation_config.json` defaults
   (LFM2.5: repetition penalty 1.1), so goldens must force pure greedy or
   they are not argmax; vapi's f32 logits at the disputed position match
   HF's f32 forward to three decimals.
2. Greedy goldens in f32 on the CPU match exactly; on the GPU in bf16 they
   match or diverge only at a near-tie, as the SmolLM2 tests define it.
   **Done.** CPU f32: all four fixtures exact for 54–64 tokens, alone and
   in a concurrent batch, cache on and off (561 s for the whole suite).
   GPU: on the RTX 5060 Ti in bf16, `capital` and `unicode` match
   all 64 tokens, `multiturn` all 54; the concurrent batch and the
   cache-on/off check pass; every divergence is at a 0.0000 logit gap.
3. Serve it through the gateway on the GPU, streaming and not; the
   prefix cache on/off check with the real model. **Done**: 22-token
   prompt, 54 ms to first token, 49 tok/s on a single stream in the
   release build with 1,024 blocks (3.3 GB of cache); the model reasons in
   a `<think>` block first because its template opens one.
4. Make free-VRAM cache sizing reachable from config (`num_blocks`
   unset), since with conv buffers the right block count depends on the
   model.

## Phase 3 — benchmark against vLLM

**Done** (`benchmark/README.md`). On LFM2.5-2.6B: parity at batch 1
(68 vs 74 tok/s, vapi's TTFT lower), then vapi's inter-token latency grows
linearly with concurrency (14.7 → 109 ms from 1 to 64) while vLLM's stays
flat (13.5 → 15.3 ms); 573 vs 3,761 tok/s at batch 64.

## Phase 4 — close the decode gap

The numbers say the per-token math is fine and the per-*sequence* work is
not. In order of expected payoff on this model, each re-measured with
`benchmark/bench.py` and gated on `just test-model` staying green.

**Progress 2026-09-21.** Item 1 (batched conv): batch-64 inter-token
latency 109 → 78 ms. Item 2 (one scatter per layer): 78 → 76 ms, within
noise, so the writes were never the cost. Then the profile the plan asked
for first, which should have come first: a batch-64 decode step is 38 ms,
of which 31 ms is the forward (8.5 ms of the host issuing kernels, 25 ms
waiting for the GPU and copying 33 MB of full-vocab logits back) and 6 ms
is host-side argmax over 64 rows of 128,000 floats; scheduling and commit
are 0.02 ms each. Items 3 and 4 below follow from that; the engine keeps
the per-phase histograms (`vapi_step_*_ms`, `vapi_forward_*_ms`) so the
next change is measured the same way. After items 3 and 4: batch-64
inter-token latency 25.5 ms and 2,305 tok/s against vLLM's 15.3 ms and
3,761; parity at batch 1 and 8. The step is 25 ms of forward (6.2 ms of
kernel issue on the host, 13.3 ms of GPU wait) and 0.07 ms of sampling.

The sampled path turned out to be the bigger problem: with Liquid's
recommended settings a batch-64 step was 335 ms, 268 ms of it the host
sampler sorting the whole vocabulary per row. Rewritten to gather only the
tokens that can carry mass (exact to f32 resolution, tested against the
old implementation) and to sample rows in place: 74.5 ms per token at
batch 64, 7 ms of sampling per step. The 33 MB logits copy (~8 ms) and two
O(vocab) host passes remain; an exact device top-k needs a custom kernel
(candle's GPU sort cannot take a 128K-wide row), so that is parked behind
the items below.

**Phase 4 is done**; its effect is in `benchmark/README.md`. Greedy and
every sampling configuration measured are at 17.8 ms per token at batch
64 against vLLM's 15.3. Not done, and only worth it if a profile says so:
capturing sampled decode steps into CUDA graphs as greedy ones are.

1. **Batched conv.** Replace the per-sequence loop in `ShortConv::forward`
   with one `index_select` per tap over the flat `(num_slots, hidden)`
   view: for every token in the batch, the slots of its own row and the
   `L - 1` rows before it are computable from the block table and position
   (`slot(p - k)`), and rows before the sequence start are masked to zero.
   Three gathers of `n` rows each per layer, no per-sequence ops, and no
   more reading a whole history to use two rows of it. Done when: the
   fixture tests still pass at every chunk split, and batch-64 inter-token
   latency drops by well over half.
2. **One cache write per layer.** `write_rows` issues a `slice_set` per run
   of consecutive slots; a decode batch is one per sequence, now for all
   30 layers. Replace with a single `scatter_set` (or `index_add` into a
   zeroed view) per layer, or a `reshape_and_cache` kernel. Done when the
   `cache.rs` tests pass unchanged and launches per step fall accordingly.
3. **Device-side sampling.** `Logits` copies 512 KB per sampled row to the
   host. **Done for greedy**: `ExecutionBackend::forward_greedy` runs
   `argmax` on the device and copies 4 bytes per row; the engine uses it
   when every sampled row is greedy and penalty-free, else the full path.
   Still to do: device top-k for sampled requests (Liquid's recommended
   settings are temperature 0.1, top-k 50, repetition penalty 1.1, which
   take the full path today), keeping the per-sequence RNG on the host.
4. **Per-step index tensors built once.** Each layer was uploading its own
   scatter index and conv tap indices, ~200 synchronous host-to-device
   copies per step. **Done**: `StepContext` builds them once per forward.
5. **Attention layers**: `.contiguous()` around RoPE. **Done**: `rope_thd`
   takes the `(t, h, d)` layout the projections produce, so both
   transposes and both copies per q and k are gone, in LFM2 and Llama;
   parity tests unchanged.
6. **Warm-up forward** at startup. **Done**: `CandleBackend::load` runs a
   one-token forward on the last block and logs its duration.
7. **CUDA graphs.** Feasibility checked against the sources: cudarc
   exposes `CudaStream::begin_capture` / `end_capture` / `CudaGraph::launch`,
   and candle allocates through cudarc's stream-ordered `malloc_async`,
   which capture accepts (allocations become graph memory nodes with
   stable addresses across replays). What the capture cannot contain, and
   the forward does today: `Tensor::from_vec` uploads from pageable host
   memory (tokens, positions, cu_seqlens, block tables, scatter and conv
   indices) and the synchronous `to_vec1` of the result. The recipe:
   - give `PagedLfm2::forward` a device-resident input struct
     (pre-allocated buffers for every per-step input, written with
     `copy_from` outside the graph) instead of building tensors inside;
   - bucket decode batches (1, 2, 4, …, 64 sequences) with padded rows
     and a bucketed `max_seqlen_k` (the flash kernel's launch geometry
     depends on it), one graph per bucket, captured lazily on first use;
   - capture forward + device argmax into a fixed output buffer; replay,
     then one D2H of `rows × 4` bytes;
   - prefill and the sampled path keep the eager route.
   Expected win: the 6 ms of kernel issue per step, so roughly 25 → 19 ms
   at batch 64 for greedy. Gate on the goldens, as always.

   **Done** (`model.cuda_graphs = true`; CUDA, greedy pure-decode
   steps; anything else stays eager, and a failed capture disables graphs
   with a warning). What the recipe above missed, found by bisecting the
   forward under capture: candle's `index_select` and `scatter_set` upload
   the index tensor's dims and strides from a temporary host buffer at
   every launch, which the capture records as a copy from freed memory and
   the replay faults on (`CUDA_ERROR_ILLEGAL_ADDRESS`). Graph memory
   nodes, casts, matmuls, RMSNorm, flash attention and elementwise ops all
   replay fine. So `rows.rs` adds two nvrtc-compiled kernels, a row gather
   and an in-place row scatter that copy raw 16-byte chunks with every
   argument by value, and the LFM2 forward uses them on CUDA for the
   embedding lookup, the conv history reads, the logits gather and the
   cache writes. Two cache blocks are reserved for padding rows.
   `tests/tiny_backends.rs::cuda_graphs_reproduce_eager_greedy_ids` is the
   regression guard; `VAPI_TEST_CUDA_GRAPHS=1` runs the 2.6B goldens
   through the graphs (they pass). Measured: batch 64 25.5 → 23.9 ms per
   token, 2,305 → 2,456 tok/s; smaller gains at 1/8/32. Less than the
   estimate because the host's kernel issue was mostly overlapped with GPU
   execution already; the remaining 23 ms per step is GPU work.
8. **Device-side candidate selection** for the sampled path. Done, as
   `fused.rs::select_candidates` (one block per row: max, per-warp
   histogram of the scaled gap in whole nats, compaction of the deepest
   bin prefix that fits 8,192 entries, plus the full softmax sum) and
   `Sampler::sample_candidates`, which applies penalties to the returned
   entries and normalises by the row's full sum corrected for what the
   penalties moved. `SamplerState::need` states the exactness condition
   per row (complete window, or `k + distinct observed` entries; a
   penalty that can raise a logit demands the complete window), and the
   backend copies the full logits of only the rows it cannot serve.
   `ExecutionBackend::forward_candidates` defaults to the full logits, so
   the mock and CPU backends are untouched. Batch 64 at the recommended
   settings: 69.6 → 17.7 ms per token.
10. **Gumbel-max draws on the device** for rows that are nothing but
   temperature and penalties (no top-k, top-p 1, min-p 0): the token is
   `argmax(logit / T + Gumbel noise)`, noise from a 64-bit hash of `(seed,
   draw index, token)`, so a seed reproduces run to run (but not the same
   tokens the host sampler would draw for that seed: the accepted trade).
   `fused.rs::gumbel_draw` applies the penalties in place first
   (`apply_penalties_f32`), then one block per row. Frequencies are tested
   against the tempered softmax within binomial noise. Temperature 0.7 and
   1.0 with no top-k at batch 64: 64 → 17.8 ms per token.

9. **Fused kernels, from a profile.** Done. `examples/decode_profile.rs`
   drives batch-N decode steps standalone; under Nsight Systems (2025.1,
   unpacked from the .deb into the scratch dir since 2024.6 cannot trace
   this driver) one cuBLAS kernel was 12 of 23 ms: the FFN gate/up GEMMs
   at M=64 land on a 16×16-tile kernel running at 217 GB/s, while the
   same-sized down projection runs at 376 GB/s. `examples/gemm_bench.rs`
   showed a GEMM of doubled width runs at 371 GB/s, so `w1`/`w3` are
   stacked at load and split by a fused SwiGLU kernel in `fused.rs`
   (nvrtc, arguments by value, capture-safe; bf16 handled as raw words so
   nvrtc needs no headers). The conv layers likewise: one `in_proj` GEMM
   and two kernels (`conv_write`, `conv_apply`) replacing three GEMMs and
   ~15 element-wise launches per layer; padded taps use a `CONV_PAD`
   sentinel index instead of mask tensors. The CPU path is unchanged and
   remains the reference. Batch 64: 23.9 → 18.6 → 17.8 ms per token,
   2,456 → 3,240 tok/s (vLLM 15.3 ms / 3,761). Weight streaming puts the
   floor near 13 ms on this card.

## Phase 5 — production behaviour and the cache tiers

Done:

1. **Engine failure policy.** A failed step fails every in-flight request
   with a `Failed` delta (the client gets an error frame, never a stalled
   stream), frees their blocks, and the engine carries on; after
   `worker.max_step_failures` consecutive failures the engine thread
   exits and the worker process with it. `Engine::fail_all`, tested with
   `MockBackend::failing_at`.
2. **Overload.** The gateway counts requests it has published that no
   worker has started (`QueueSlot`, released on `Started`); past
   `gateway.max_queued_requests` it answers 503 with `Retry-After: 1`
   without touching NATS. `Error::Overloaded` moved from 429 to 503 with
   code `overloaded`.
3. **Cancellation and preemption on the real model.** `real_model.rs`
   cancels one of two live requests and checks the other matches its
   golden and no block leaks; and sizes the cache two blocks short of two
   long generations so one is preempted, then checks both recompute to
   their goldens exactly.
4. Step-time histograms: done in phase 4.
5. **Graceful drain.** SIGTERM or Ctrl-C stops the worker pulling jobs,
   keeps heartbeats and cancels flowing, exits when in-flight work is
   done, and fails what is left at `worker.drain_timeout_secs`
   (`Command::FailAll`).
6. **logprobs.** Rows that ask are excluded from every device shortcut
   and read off the full row before the sampler touches it
   (`RowLogprobs`); `Delta::Token` carries `logprob` and `top_logprobs`;
   the gateway emits OpenAI's chat shape (streaming and not) and the
   legacy completions shape.
7. **Tier 2 response cache** (`vapi-gateway/src/response_cache.rs`):
   NATS KV bucket `nats.response_cache_bucket` with
   `cache.response_cache_ttl_secs`; key = blake3 of weight fingerprint,
   job kind, prompt ids, every sampling field, `max_tokens`, stop
   strings; populated only for deterministic requests without logprobs
   that finished by stop or length; hits are answered by the gateway,
   replayed delta by delta when streaming.

8. **Tool calling and reasoning.** `tools` and `tool_choice` on the chat
   request; the tool definitions and any `tool_calls` / `tool` messages go
   through the model's own chat template (`render_with(..., tools)`),
   which needed a Python-compatible `tojson` filter and insertion-ordered
   maps in minijinja, since `json.dumps` spacing and key order are part of
   the prompt. `vapi-gateway/src/output.rs` parses the answer back:
   `</think>` splits reasoning from content, and
   `<|tool_call_start|>[name(arg='v')]<|tool_call_end|>` is parsed
   (Python call syntax, with JSON literals, numbers, `True`/`False`/`None`)
   into `tool_calls` with JSON arguments, stable ids, and a `tool_calls`
   finish reason. Unparseable calls are passed through as text. Two new
   goldens (`tools`, `toolresult`) check the rendered prompt against
   Hugging Face byte for byte.

9. **Tier 3 spill.** `vapi-cache/src/spill.rs` holds evicted blocks in
   host memory with a disk tier behind it, both LRU, keyed by the same
   content hash the prefix cache uses (so entries cannot cross models and
   survive a restart). `BlockPool::take_evicted` reports what the
   allocator dropped; the engine exports those blocks between scheduling
   and the forward, which is the only window before they are overwritten,
   and restores a prompt's prefix at admission. `ExecutionBackend` grew
   `block_bytes` / `export_block` / `import_block`; `MockBackend`
   implements them so the path is tested without a GPU.
   **The measurement says leave it off by default**: restoring a block is
   2.4 ms against 4.1 ms to recompute, but each eviction pays 2.7 ms to
   export, so it needs ~1.5 restores per spilled block to break even.
   Pinned host buffers are the obvious next step, and would change that.
10. **Multi-worker cache affinity.** `nats.job_partitions` partitions the
   job subject; the gateway routes on a hash of the prompt's **first
   block**, so every turn of a conversation routes the same way as it
   grows (hashing the whole prompt would scatter them). The hash is FNV
   plus a mixing round, pinned by a test, because gateway and worker have
   to agree across releases. Each worker serves `worker.partitions` with
   one durable consumer per partition and merges their streams, and
   publishes itself into the `VAPI_WORKERS` KV bucket
   (`vapi-worker/src/registry.rs`) for observability only: routing never
   reads it, so a registry outage cannot misroute anything. Measured on
   two workers with eight concurrent conversations: prefix hit rate 0.40
   → 0.49, load 20/20 → 15/25.

**Phase 5 is complete.**

## Phase 6 — what a production user hits next

Order is by value on this hardware, not by size.

1. **Structured output.** Done. `vapi-engine/src/structured.rs` compiles
   the schema subset into an arena and walks it with a cheap-to-clone
   machine: a token is allowed when feeding its text to a copy leaves the
   document both valid *and* still completable, which is what stops the
   model walking into a corner (a comma in an object whose every property
   has appeared). `TokenMasker` buckets the vocabulary by first byte so a
   step tries a few thousand tokens rather than 128K. Constrained rows
   take the full-logits path, as `logprobs` rows do, since a mask can rule
   out everything the device selected.

   Three things the model made necessary, none of them obvious up front:
   LFM2.5 drafts its answer inside its reasoning block and then stops, so
   a constraint that simply waits for `</think>` gets nothing; the fix is
   to make the model emit the closing marker when it tries to end its
   turn, and to force it once thinking has taken three quarters of the
   budget. And the last `CLOSING_BUDGET` tokens only allow what closes the
   document, so a tight budget yields valid JSON instead of a truncated
   value.

   Found on the way: the tier-2 response cache keyed on the prompt and
   sampling parameters but **not** on `response_format`, so a request
   asking for JSON was served an earlier plain answer to the same prompt.
   Fixed, with a test.
2. **A second real architecture.** Done: Qwen3-0.6B, in
   `models/qwen.rs`, covering Qwen2 as well. Three things separate it from
   the paged Llama and each is a way to be silently wrong: `head_dim` is
   its own number (Qwen3-0.6B has 16 heads of 128 over a hidden size of
   1024), Qwen2 biases Q/K/V while Qwen3 does not, and Qwen3 normalises Q
   and K per head before the rotary embedding. Proven first on a tiny
   random fixture against `transformers` (`tools/gen_qwen_goldens.py`,
   with a deliberately non-derivable `head_dim`), then on the real
   checkpoint: all four chat goldens match token for token in f32 on the
   CPU, and tie-only divergences in bf16 on the GPU.

   The gateway learned Qwen's dialect too: it writes its own `<think>`
   opener (LFM2's prompt supplies one) and its tool calls are JSON inside
   `<tool_call>` tags rather than Python call syntax. `OutputFormat` now
   carries both markers and a syntax, picked from the vocabulary.

   Throughput, unconstrained greedy: 191 tok/s at batch 1 and 5,937 at
   batch 64, with CUDA graphs off; phase 7 turned them on for Qwen and
   took it to 7,325.
3. **`n > 1` with copy-on-write forking.** Done. The prompt is prefilled
   once; at the step where the leader produces its first token, the
   scheduler forks `n - 1` siblings that share the prompt's full blocks by
   reference, and each choice draws its own first token from that same
   logits row. `Scheduler::fork` returns the copy pairs for the backend.

   The subtlety is which block is private. Blocks the prompt fills
   completely are shared; the block after them is open, and **whether or
   not it holds any prompt tokens**, every choice is about to write a
   different token into it, so each gets its own. Sharing an empty open
   block is the half that is easy to miss, and the test for a
   block-aligned prompt is there to catch it.

   Rows about to fork ask for their full logits, because the device
   shortcuts return a single drawn token or a candidate set, and neither
   can be sampled from again. Greedy requests are not forked: the choices
   would be identical. `n > 1` is excluded from the response cache, whose
   entries hold one completion. A fork that cannot be made closes the
   choices it will not serve, so a caller never waits for completions that
   are not coming.
4. **Quantisation.** Done, via GGUF: `PagedQwen::load_gguf` reads
   llama.cpp's tensor names and metadata, and the weights stay quantised
   in candle's `QMatMul`. A directory holding a `.gguf` is taken as a
   quantised model; its tokenizer still comes from the directory, since
   the gateway and worker share one tokenizer and chat template.

   Two things worth keeping: the first version converted activations to
   f32 around every projection, because a quantised matmul against bf16
   looked like a dtype error. The error was the *bias*, not the matmul;
   candle's kernels take bf16 directly. Casting the bias once at load
   instead took batch-1 decode from 7.7 ms to 4.5 ms. And norm weights
   are one vector per layer, so they are dequantised once at load rather
   than converted every step.

   Measured on a 16 GB card: Qwen2.5-7B-Instruct Q4_K_M runs in 8.3 GB
   and answers correctly, where bf16 would need 14 GB of weights before
   any KV cache. Quantisation is *faster* than bf16 at batch 1 (4.5 ms
   against 5.3 on the 0.6B) and slower above it. `benchmark/README.md`
   has the numbers and the candle batch-8 cliff.

**Phase 6 is complete.**

## Phase 7 — carry the optimisations across

Done. The phase-4 work was written for LFM2; this made it general.

1. **The capture path is a trait.** `graphs::Capturable` asks a model for
   a `prepare` (host work: every device tensor a step needs) and a
   `forward_prepared` (device work only, no host traffic), plus how to
   copy inputs in place. `GraphRunner` is generic over it and the backend
   holds whichever runner the loaded architecture needs. LFM2 had this
   split already; Qwen grew one.
2. **Qwen's step is capture-safe.** The embedding lookup and the logits
   gather use `rows::gather` rather than `index_select`, the KV writes use
   `rows::scatter` rather than `write_kv_to_cache`, and attention takes a
   `PreparedAttention` instead of rebuilding its descriptors from the host
   batch each step. Those are the three places candle uploads shape
   metadata from a temporary host buffer, which a capture records as a
   copy from freed memory.
3. **The fused FFN is shared.** Qwen's gate and up projections are stacked
   at load into one GEMM and split by the fused SwiGLU kernel, as LFM2's
   are. The quantised path keeps them apart: a `QMatMul` holds packed
   integer blocks, not a tensor to concatenate.

Device-side sampling needed no work: the candidate selection and the
Gumbel draw were written against the logits tensor, so they already
applied to any architecture.

Measured on Qwen3-0.6B: 6,058 → 7,325 tok/s at batch 64, 186 → 235 at
batch 1, and the gap to vLLM closed from 1.5-1.65x to 1.17-1.36x. LFM2.5
is unchanged, which is the other half of the result.

## What the benchmark says next

The earlier reading, kept because it is what motivated phase 7: on LFM2.5, the
architecture phase 4 tuned, vapi is level with vLLM at batch 1, ahead at
8 and 1.15x behind at 64. On Qwen3-0.6B, which runs the plain path, vLLM
is 1.5 to 1.65x ahead. The difference between those two numbers is the
phase-4 work, and none of it is architecture-specific in principle: CUDA
graph capture, the fused FFN, and device-side sampling would all apply to
Qwen. Carrying them across is the obvious phase 7, and it is porting
rather than research.

## Standing rules

- Golden parity before performance, every time.
- The CPU path stays the numerical reference.
- Nothing lands for the real model that was not first proven on the tiny
  fixture against transformers.
- `benchmark/bench.py` runs after any change to the step, and the JSON is
  committed.
