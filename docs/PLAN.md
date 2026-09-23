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

Phases 6 and 7 widened that destination to Qwen2/Qwen3 and to quantised
weights. Phase 8 leaves the decoder altogether: a bidirectional encoder
that answers in one forward pass, which is a second execution path rather
than one more architecture. Phase 9 goes further still and adds an input
modality — speech, arriving over time.

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

## The dashboard

`/dashboard` in the gateway: server-rendered HTML, plain form posts, one
small script that refreshes the counters. It reads the worker registry
bucket for the per-worker view, keeps its own exact counters rather than
scraping the metrics exporter back, and holds the last 25 finished
requests in a ring buffer.

The line it draws: settings that live in the gateway process (queue limit,
the two stream timeouts, whether the response cache is consulted) are
editable and take effect on the next request; everything a worker reads at
its start is shown read-only. Changes are not written back to `vapi.toml`.
The template is registered as `.html` so minijinja escapes by default,
since everything on the page comes from a request, a worker or a form.

Not done: authentication. It is a local operator page today.

## Phase 8 — models that answer in one pass

Done. A second execution path: a bidirectional encoder that answers a whole
request in one forward pass, with `convaiinnovations/laya` as the first one.
`POST /v1/decisions` takes a state and typed questions and returns calibrated
probabilities. The point is not the one model — the same path is what
embeddings, cross-encoder rerankers and classifier guardrails would use, none
of which vapi could do at all before.

### What the model is

A ModernBERT-large encoder (28 layers, d=1024, 16 heads, sliding attention of
128 on two layers in three, RoPE theta 160k global and 10k local) plus a head
trained from scratch: a type embedding added at every position, two pre-LN
transformer layers with a ReLU feedforward, a scorer read at each option's
`[MASK]` marker, and an act head. 421M parameters, 842 MB.

Each question becomes its own sequence:

```
[CLS] choice question: <instructions> [SEP]
[MASK] billing: invoices, payments, refunds [MASK] technical: bugs ... [SEP]
<serialized state> [SEP]
```

The scorer reads the marker positions and softmaxes over that question's
options, after dividing by a temperature fitted per (type, option count). The
answer space is defined per request, so a new schema needs no retraining.

### What landed

1. **A second trait, not a wider `ForwardBatch`.** `EncoderBatch` carries
   tokens, marker positions and a question type per row; `EncoderBackend`
   returns marker logits. `MockEncoder` plays the part `MockBackend` plays for
   decode, and validates every batch it is handed.
2. **`models/modernbert.rs`**, written against a **flat, unpadded layout** —
   every row's tokens end to end with cumulative lengths — and
   `flash_attn_varlen_windowed(causal = false, left = 64, right = 64)`, which
   is what `local_attention: 128` means. The first version was padded-dense
   and measured 2.5 s for 64 rows of 512 tokens, because a batch then costs
   `rows × longest` whatever the rows are and the attention it implies
   materialises 537 MB of padding arithmetic per layer.
3. **`models/laya.rs`**: the head, on the same flat layout, with the markers
   gathered in one `index_select` over `[total, hidden]`.
4. **A decision engine beside the autoregressive one**, not inside it. No
   collection window and no artificial delay: batching emerges under load
   exactly as it does on the decode path, because arrivals queue behind the
   running pass and the next pass takes them all.
5. **Proto**: `JobKind::Decision`, a terminal `Delta::Decided`, and
   `Job::rows` describing how the prompt divides. Carried beside the tokens
   rather than inside the kind, so the partition hash, the token accounting
   and the cache key keep working without knowing decisions exist.
6. **Gateway**: the sequence builder, the calibration, and the answer shaping,
   all of it where the chat template already is. An unanswerable request is
   refused before any queue work.

The worker picks its engine once, at startup, from the checkpoint's layout: a
model is either a decoder or an encoder, never both.

### Parity

The standing rule held. A tiny random ModernBERT plus head
(`~/models/laya-tiny`, written by `tools/gen_laya_goldens.py --tiny`) came
first, then the real checkpoint. Sequences are checked token for token against
the reference builder, and answers against the reference model:

| check | English | multilingual |
| --- | --- | --- |
| sequences vs the reference builder | exact | exact |
| encoder hidden states, CPU f32 vs transformers | 2.5e-5 | 3.8e-5 |
| answer logits, CPU f32 | 7.6e-6 | 1.1e-5 |
| answer probabilities, CPU f32 | 4.6e-5 | 4.9e-5 |
| answer probabilities, CUDA bf16 vs CPU f32 | 1.5e-2 | 1.3e-2 |

The argmax is unchanged everywhere, including in bf16. The multilingual
column is the bundled mmBERT-base checkpoint, run on fixtures in Hindi,
Khmer, Cyrillic and Japanese as well as the English ones — a different
encoder shape (22 layers of 768, 256k vocabulary), a different tokenizer
(Metaspace rather than byte-level, with `<bos>`/`<eos>`/`<mask>` for the
structural tokens) and a 1024-token budget, so it exercises the loader rather
than repeating the English run. Its `position_embedding_type: "sans_pos"` is
inert: transformers' ModernBERT never reads that field.

Two fidelity bugs were found by those goldens rather than by a user.
`json.dumps` separates with `", "` and `": "` where serde's compact writer
uses `","` and `":"`, which is five tokens of difference on the email fixture
— a different sequence than the one the model was calibrated on. And the
scorer reads specific positions, so the builder's budget arithmetic has to
match the reference exactly, down to the 48-token cap on an option.

A third was found by the CUDA path: the kernel choice and the window mask were
derived independently, and disagreed, so a batch could take the reference path
with no mask and attend over everything. They are one decision now, stored on
the prepared batch. The test that caught it compares the kernel against the
reference **at the same dtype**, which is the only comparison that separates a
wrong window from half-precision drift — and half-precision drift here is
large, because ModernBERT carries activation outliers of magnitude ~25 that
the scorer's own LayerNorm removes before they reach an answer.

### Measured

RTX 5060 Ti, bf16. Forward pass only
(`cargo run --release --features cuda --example decision_bench`):

| rows | tokens each | ms | ms/row |
| --- | --- | --- | --- |
| 1 | 96 | 9.5 | 9.49 |
| 16 | 96 | 67.4 | 4.21 |
| 64 | 96 | 288.2 | 4.50 |
| 1 | 512 | 25.8 | 25.76 |
| 64 | 512 | 1584.4 | 24.76 |

End to end, through HTTP and NATS (`benchmark/bench_decisions.py`):

| checkpoint | 1 question | 4 | 16 | 32 clients |
| --- | --- | --- | --- | --- |
| English (ModernBERT-large) | 10.2 ms | 20.5 ms | 69.4 ms | 260 questions/s |
| multilingual (mmBERT-base) | 8.3 ms | 10.5 ms | 34.5 ms | 536 questions/s |

The published reference is a Tesla T4 at 39.5 ms for one question and 103-332
questions/sec batched. The 2.1x between the two checkpoints is the smaller
encoder, and matches the 2.2x their own benchmark reports.

Two things that reading those numbers should not miss. **Cost is proportional
to the tokens actually present**, not to `rows × longest`: 64 rows of 96
tokens and 12 rows of 512 take the same time, which is the whole point of the
unpadded layout. And **batching buys little** — 4.5 ms/row at 64 rows against
9.5 ms alone — because the pass is compute-bound at about 21,000 tokens/s and
a question is only ~90 tokens. The ~5 ms floor on a single question is kernel
launch overhead across 28 layers, which is what CUDA graph capture fixed for
decode in phase 6 and would fix here; it is the obvious next optimisation and
was not done.

### The router

The English checkpoint collapses on non-Latin scripts *while staying
confident* — the published figure is 0.000 accuracy at 0.952 confidence on
Khmer — so no confidence threshold downstream can catch it. `vapi_core::script`
reads a text's dominant script in microseconds, and the gateway warns and
counts `vapi_decision_unreadable_script_total` when the loaded checkpoint
cannot read what it was sent. Whether the checkpoint is multilingual is a
heuristic on the encoder's vocabulary size, since nothing in the checkpoint
states it.

Both checkpoints run: point `model.path` at the repo root for English or at
its `multilingual/` subfolder for the other, and the layout check finds
either. The difference is worth seeing on one input. Given
`मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।`:

| checkpoint | department | confidence | refund requested |
| --- | --- | --- | --- |
| English | billing | **0.06** | — |
| multilingual | billing | **0.95** | 0.996 |

The English one happens to land on the right answer here while reporting that
it is guessing; on Khmer the published figure is the opposite failure, 0.000
accuracy at 0.952 confidence.

What is **not** done: serving both *at once* and dispatching between them per
request. That needs a tokenizer, a budget and a calibration table per
checkpoint in the gateway, because they differ in all three, and the sequence
has to be built with the tokenizer of whichever checkpoint will answer. The
partitioned-subject affinity from phase 5 is the right mechanism for the
worker side — one checkpoint per worker, the gateway choosing by detected
script — and the detection it needs is now there. Until then, two vapi
deployments serve the two checkpoints.

### Router follow-ups

Notes for whoever picks this up, in the order the obstacles actually appear.

**The routing decision has to happen before tokenization.** The gateway builds
the sequence, and it must build it with the tokenizer of the checkpoint that
will answer — the two differ in vocabulary, in `max_len`/`head_max_len`, and
in whether an option gets truncated. Script detection already runs on the raw
state text, so the ordering works out; anything that needs the tokens to
decide would not.

**What has to become plural.** `AppState` holds one `DecisionFormat`, one
`ModelId` and one weights fingerprint. All three are per checkpoint: the
fingerprint is in the response-cache key, and sharing one across checkpoints
would serve one model's answer for another's request. `prepare()` takes the
model id from state; it would take the chosen one.

**Publishing is already solved.** Each worker registers under its own model
id and consumes `vapi.jobs.<model>`, so the gateway routes by publishing to
the chosen checkpoint's subject. Nothing in the worker, the engine, the proto
or the registry changes — one checkpoint per worker is what they already do.
The partitioned-subject machinery from phase 5 is not needed for this: it
partitions *within* a model for cache affinity, and there is no cache here.

**Config shape.** `[model]` is singular throughout. The smallest change that
works is an array — `[[decision.checkpoints]]` with `path`, an optional
`name`, and which scripts it claims — rather than overloading `model.path`.
Resident cost is about 1.8 GB for all three, which the 16 GB card does not
notice.

**Let a request name its checkpoint.** The published API takes
`model="typed-decisions"` as an explicit override, and the third checkpoint is
a fine-tune rather than a script variant, so it can only be reached that way.
Script detection should be the default, not the only path.

**Say which one answered, and why.** The published router returns the
checkpoint, the repo and a reason string (`"non-Latin script (devanagari,
100% of letters); the English checkpoint cannot read it"`). `DecisionResponse`
carries `model` already; a `routing` object beside it is the honest version,
because a caller comparing two answers needs to know they came from different
weights.

**The trap.** Probabilities from two checkpoints are not comparable. The
English one ships a fitted temperature table and the multilingual one ships
none, so a threshold tuned on one silently means something else on the other.
Whatever the router does, it should not let that difference go unstated —
per-checkpoint calibration is the thing that makes a routed answer mean the
same as an unrouted one, and refitting is the operator's job.

**The cheap alternative, honestly.** Two vapi deployments and a reverse proxy
doing the script check is most of the value for none of this work. It is the
right answer if the routing is the only reason to want multiple checkpoints;
the case for doing it inside vapi is one process, one dashboard, one set of
metrics, and the request naming its own checkpoint.

### Honest limits

- The **base checkpoint is near chance zero-shot** on the typed-decisions
  benchmark: 0.362, against a 0.461 majority-class baseline. The 0.766
  headline belongs to a checkpoint fine-tuned on that benchmark's own training
  split. Laya is a fast base to specialise, not a zero-shot decision engine.
- **Cramped options are answered and reported, not refused.** Below about
  eight tokens each, options stop being distinguishable and accuracy falls
  from 0.870 to 0.425 on a 77-option benchmark. The builder refuses only when
  an option cannot fit at all, and logs when it had to cut; a request that
  fits is never silently degraded without a warning.
- **The response cache is off for decisions.** They are pure functions of
  their input and would cache perfectly, but the cache stores generated text.
- **A decision worker has no KV cache and no prefix cache**, and the dashboard
  shows zeroes for both. That is not a gap to fill: a prefix cache would be
  *wrong* here, because attention is bidirectional and a prefix's
  representation depends on the state that follows it.
- The shipped temperatures are fitted on the publisher's data. Mean ECE is
  0.466 before scaling and 0.081 after, so refit them on yours. **The
  multilingual checkpoint ships no fitted table at all** — all three
  temperatures are 1.0 and the bucket table is empty, so its probabilities are
  raw. Its published raw ECE is 0.314.

## Phase 9 — speech in

Started. The frontend is done and proven; the model port is not. A third
input modality: audio arriving over time, with
text coming back as it is spoken. `mistralai/Voxtral-Mini-4B-Realtime-2602` is
the target — Apache 2.0, 4B parameters in bf16, and the first realtime
transcription model that fits this card.

### What the model is

Two stacks and a projector, and the pleasant surprise is that **neither stack
is a new attention shape**.

| | audio encoder | text decoder |
| --- | --- | --- |
| parameters | ~970M | ~3.4B |
| layers | 32 | 26 |
| hidden | 1280 | 3072 |
| heads | 32 × 64 (MHA) | 32 × 128, 8 KV (GQA) |
| feedforward | 5120, SiLU | 9216, SiLU |
| attention | causal, sliding window 750 | causal, sliding window 8192 |
| rope theta | 1e6 | 1e6 |
| norm | RMSNorm, eps 1e-5 | RMSNorm, eps 1e-5 |

Both are Llama-shaped with an explicit `head_dim` that is not
`hidden / heads` — which `qwen.rs` already handles — and both are causal with
a sliding window, which `flash_attn_varlen_paged_windowed` already does and
`laguna.rs` already uses.

**What reading the checkpoint corrected.** The tensor map (711 tensors) says
three things the config does not:

- **Every decoder layer carries an adaptive RMSNorm** — `ada_rms_norm.linear1
  [32, 3072]` and `linear2 [3072, 32]`, applied as
  `hidden_states * (1 + linear2(gelu(linear1(t_cond))))`. `t_cond` is a
  parameter-free time embedding of `num_delay_tokens`: there is no
  `time_embedding` tensor in the checkpoint. This is how one set of weights
  serves every delay from 80 ms to 2.4 s, and it is cheap — the conditioning
  does not vary by position or step, so the 26 scale vectors are computed
  once per session and then are free. The earlier claim here that the decoder
  is "`qwen.rs` with tied embeddings" was wrong.
- **The audio encoder follows Whisper's bias convention**: biases on `q_proj`,
  `v_proj` and `o_proj` but **not** on `k_proj`, and on `mlp.down_proj` only.
  Its MLP is gated (gate/up/down), unlike Whisper's.
- **The projector groups frames rather than striding.** `linear_1` is
  `[3072, 5120]` and 5120 is 1280 × 4: four consecutive encoder frames are
  reshaped into one vector and projected. With `conv2`'s stride of 2 on top,
  that is the 8 mel frames — 80 ms — per decoder position.

The conv stem is two causal `kernel_size=3` convolutions, the second with
stride 2, each carrying a padding cache across streaming chunks. The
genuinely new pieces are that stem, the projector, the adaptive norm, and the
frontend that produces the mel bins at all.

### The mechanism, which decides everything else

Audio is 16 kHz mono; the frontend is Whisper-shaped (`n_fft` 400,
`hop_length` 160, 128 mel bins), so one mel frame is 10 ms.
`audio_length_per_tok: 8` and `downsample_factor: 4` put **one decoder
position on every 80 ms of audio**, and the decoder emits exactly one text
token per position. That is what "a single text-token is worth 80 ms" and
"exceeding 12.5 tokens/second" mean.

The merge is one line of the reference: `inputs_embeds += audio_embeds`. The
audio embedding is **added to the text token embedding at the same position**,
not interleaved as a position of its own. So a step's input is
`embed(previous text token) + audio_embed(this 80 ms frame)`, and
`default_num_delay_tokens: 6` is how many frames the audio runs ahead of the
text — 480 ms, their recommended setting.

Two consequences worth stating before any design. **Decoding is lock-step with
wall-clock audio**: not "generate until EOS" but exactly one step per 80 ms,
forever, and a session that has no audio yet must not be stepped. And **the
prompt never ends**, which is the assumption `Sequence` is built on.

### Why it does not fit, in order of how much it hurts

1. **A sequence whose prompt keeps growing.** `Sequence` takes a complete
   prompt at admission and prefills it. Here admission is the start of a
   conversation that may run for three hours. It needs a state the scheduler
   understands as "runnable only when its audio has arrived", distinct from
   waiting on KV blocks.
2. **Embeddings as input.** `ForwardBatch.tokens` is `Vec<u32>`; there is no
   way to say "and add this vector at this position". This is the largest
   interface change, and it is the one to design first because everything
   else is downstream of it.
3. **A second model with streaming state.** The encoder carries its own KV
   (bounded by its 750 window) *and* a conv1d padding cache across chunks, so
   a chunk boundary is not a clean edge. Both must be per session and both
   must be freed when it ends.
4. **Real-time pacing.** The engine steps as fast as it can. A transcription
   session steps 12.5 times a second and must not run ahead of its audio.
   This is good news for throughput — many sessions batch into one step
   naturally — but the loop has to learn to wait.
5. **Transport.** Audio is a byte stream, not a request. NATS' default
   `max_payload` is 1 MB and an hour of 16 kHz mono PCM is 115 MB, so audio
   cannot ride the job queue as it stands. A live session wants a
   worker-affine channel, not a durable queue; the registry and partitioned
   subjects give the affinity, JetStream's object store gives the durable
   option for whole files.
6. **The tokenizer.** The repo ships `tekken.json` and **no** `tokenizer.json`.
   `vapi-tokenize` requires the latter and says so plainly, which is the right
   behaviour and also a blocker. Either convert once at load, or teach it
   Tekken (a tiktoken-style BPE: base64 vocabulary, a split regex, a special
   token list).
7. **Sliding-window eviction.** Past 8192 positions — about 11 minutes of
   speech — every block below the window is dead. A three-hour meeting is
   only affordable if those blocks come back to the pool. The pool is
   refcounted and the kernel already takes a window; what is missing is the
   scheduler dropping block-table entries the window has passed.

### What gets reused

Continuous batching, the paged block pool, the windowed paged kernel, CUDA
graph capture, the registry, the dashboard, metrics, the failure and drain
policy. `LayerLayout` already lets one paged cache hold layers whose KV shapes
differ, which is exactly what two stacks in one worker need. The text decoder
is `qwen.rs` with tied embeddings; the audio encoder is the same block with a
conv stem in front.

The `EncoderBackend` from phase 8 does **not** apply. That trait is for models
that answer in one pass; this is a decoder with a paged cache and a step loop,
which is `ExecutionBackend` with a wider input.

### What decides capacity

Worth doing before writing code, because it sets the shape of the product.

- weights: 8 GB bf16, leaving about 7 GB on a 16 GB card
- audio encoder KV, per session, bounded by its 750 window:
  `32 × 750 × 32 × 64 × 2 × 2 B` = **196 MB**
- text decoder KV: `26 × 8 × 128 × 2 × 2 B` = **104 KB per token**, and a full
  8192 window is **852 MB**

A window only fills after 8192 × 80 ms ≈ 11 minutes, so the realistic figure
is per call length: a two-minute call holds 1,500 text positions (156 MB) plus
196 MB of encoder KV, about **350 MB**, so roughly **20 concurrent
two-minute calls** on this card. Hour-long meetings are about six. The encoder
KV is the surprise — it is fixed per session and larger than the text cache
for any call under two and a half minutes.

Measured against the reference, that arithmetic holds: weights take 8.25 GiB
in bf16 and a 60-second session adds 0.33 GiB, against the 0.27 GiB the sum
above predicts.

### The reference baseline

`benchmark/bench_transcription.py`, transformers on the RTX 5060 Ti in bf16,
against real speech repeated to length:

| audio | wall | realtime | steps/s | peak |
| --- | --- | --- | --- | --- |
| 1.4 s | 0.88 s | 1.59x | 19.3 | 8.34 GiB |
| 10 s | 6.49 s | 1.54x | 19.3 | 8.47 GiB |
| 30 s | 19.2 s | 1.56x | 19.5 | 8.52 GiB |
| 60 s | 38.9 s | 1.54x | 19.3 | 8.58 GiB |

The number that matters is **19.3 steps a second against the 12.5 a live
session consumes**: one stream keeps up with 54% to spare, and that margin is
flat from 1.4 seconds to a minute. Two things follow.

Memory is not the binding constraint — about twenty 60-second sessions fit in
the 7.5 GiB left over — but **compute is**: one stream at a time already uses
two thirds of the card, so a second concurrent session does not fit without
batching their steps together. That is precisely what the engine here does
for decode and what the reference implementation does not do at all, and it
is the strongest argument for the port. It is also the number phase 9 should
be judged on: not tokens per second, but how many concurrent sessions hold
1.0x.


### Progress

Done and proven against the reference:

1. **Tekken.** `tekken.json` is a tiktoken-style BPE — a split regex, 150,000
   ranked byte strings and a 1,000-entry special block in front — and the repo
   ships no `tokenizer.json`, so `vapi-tokenize` reads the format directly.
   Two details decide the id arithmetic and both are pinned by tests: an entry
   of rank `r` is id `r + 1000`, and the vocabulary is truncated to
   `vocab_size - 1000` = 130,072 of its 150,000 entries. All 20 golden
   strings — thirteen languages, emoji with zero-width joiners, whitespace
   runs, and the spellings of the model's own control tokens — encode and
   decode identically to `mistral-common`.
2. **The frontend.** A new `vapi-audio` crate: dependency-free, tensor-free
   STFT and mel filterbank. Matches the reference feature extractor to 1e-4,
   which is the goldens' own rounding precision.

   Two things had to be right and were not at first. The triangles are laid
   out **in Hz, not in mel** — both conventions exist in the reference library
   and laying them out in mel space moved every bin by up to 2.1 against a
   total range of 2. And the DFT accumulates **in f64**: high-frequency speech
   energy sits orders of magnitude below the fundamental, so an f32 sum of 400
   terms lost it to cancellation, leaving the bottom of the filterbank exact
   and the top 2.6% out. The log floor is a global constant rather than the
   utterance's own maximum, which is what makes the frontend streamable at
   all.

3. **The reference, pinned.** `tests/goldens/voxtral-model.json` records one
   clip of real speech end to end: the prompt the processor builds (`<s>`
   followed by 38 `[STREAMING_PAD]` — 32 left-pad tokens plus the 6 delay
   tokens plus one), the shapes at every stage, four full rows each of the
   encoder output and the projected audio embeddings, the time-conditioning
   vector, the top-8 logits at 24 positions, and the transcript. A synthetic
   tone was the first fixture and was a bad one: the model correctly answers
   it with nothing but padding tokens, so every argmax agrees whatever the
   port does. Real speech with a known transcript replaced it.

Not started: the two model files, the embedding side-channel, the transport
and the endpoint. Every unknown the plan flagged is now resolved and recorded
above, so what remains is writing them.

### How it lands

**9a — offline transcription first.** `POST /v1/audio/transcriptions` in
OpenAI's batch shape: upload a file, get text back. This proves the entire
model stack — frontend, encoder, projector, additive embeddings, lock-step
decode — with none of the session lifecycle, because a complete file is a
complete prompt. Files over about 1 MB need the object store or a chunked
publish, which is the only transport work in it.

1. Tekken, or a conversion to `tokenizer.json`, proven against
   `mistral-common` on a fixture.
2. The mel frontend against the reference feature extractor, sample for
   sample. candle's Whisper example is the nearest reference.
3. `models/voxtral_audio.rs` and `models/voxtral_text.rs`, each against a tiny
   random fixture first and then the real weights, as the standing rule
   requires. `transformers >= 5.2` has
   `VoxtralRealtimeForConditionalGeneration`, so goldens generate the same way
   phase 8's did. Both unknowns are now read out of the reference and
   recorded here. The time embedding is sinusoidal and parameter-free:
   `inv_freq = exp(-ln(10000) * arange(1536) / 1536)`, then
   `cat(cos(t * inv_freq), sin(t * inv_freq))` for the scalar
   `t = num_delay_tokens`. The adaptive scale is applied **between the
   post-attention norm and the MLP**, as
   `h = post_attention_layernorm(h) * (1 + linear2(gelu(linear1(t_cond))))`,
   with the residual taken before the norm. The encoder's stem is
   `gelu(conv1)` then `gelu(conv2)`, both causal with left padding of
   `kernel - stride`, and the projector is `linear_2(gelu(linear_1(x)))` over
   four grouped encoder frames, all three without biases.
4. The embedding side-channel through `ForwardBatch`, and the conv/KV
   streaming caches.
5. The endpoint, and a word-error-rate check on a public clip rather than
   only a byte-parity check — parity proves the port, WER proves the product.

**9b — realtime.** A WebSocket session on a subset of OpenAI's realtime
transcription surface, a session-affine channel to one worker, real-time
pacing in the engine, sequences whose prompt keeps growing, and
sliding-window block eviction. Partial results as they are spoken.

### Honest limits

- **13 languages**, and the model card says to keep temperature at 0.
- **No timestamps** in the output. Anything wanting word-level timing needs a
  different model or alignment on top.
- OpenAI's realtime API is a large surface — session events, turn detection,
  interruption. A subset is the honest goal, and the subset should be named in
  the docs rather than implied.
- This is **bigger than phase 8**, which reused the whole transport and added
  one trait. Phase 9 adds an input modality, two model files, a new input type
  through the backend interface, a new sequence lifecycle and a new transport.
  9a is comparable to phase 8; 9b is not.

### What would settle whether it is worth it

Not tokens per second. A session must sustain 12.5 tokens/second or it falls
behind the speaker, so the only question that matters is **how many concurrent
sessions stay real-time**, with the answer stated at a call length. vLLM's
realtime API serves the same model and is the comparison, as it has been since
phase 3.

## Standing rules

- Golden parity before performance, every time.
- The CPU path stays the numerical reference.
- Nothing lands for the real model that was not first proven on the tiny
  fixture against transformers.
- `benchmark/bench.py` runs after any change to the step, and the JSON is
  committed.
