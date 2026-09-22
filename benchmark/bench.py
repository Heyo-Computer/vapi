#!/usr/bin/env python3
"""Load benchmark for an OpenAI-compatible chat endpoint.

Runs N concurrent streaming chat completions for each batch size, all
distinct prompts, greedy, and reports time-to-first-token, inter-token
latency and aggregate throughput measured at the client. Used to compare
vapi and vLLM on the same model and GPU; run it against each server in turn
so they never share the card.

    python benchmark/bench.py --url http://localhost:8080 --model m --label vapi
    python benchmark/bench.py --url http://localhost:8001 --model smollm2 --label vllm

Writes a JSON record per run to benchmark/results/<label>.json and prints a
markdown row per batch size.
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    f"Write a short paragraph about topic number {i}: the history of a European city."
    for i in range(128)
]


SAMPLING = {"temperature": 0}


def one(url, model, i, max_tokens, out):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPTS[i]}],
        "stream": True,
        "max_tokens": max_tokens,
        **SAMPLING,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    stamps = []
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            # A reasoning model streams its thinking as `reasoning_content`
            # before any `content`. Those are tokens too, and a run that
            # counted only content would report no tokens at all for a
            # short budget spent thinking.
            delta = d["choices"][0]["delta"] if d["choices"] else {}
            if delta.get("content") or delta.get("reasoning_content"):
                stamps.append(time.time())
    out[i] = (t0, stamps)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def run_batch(url, model, batch, max_tokens):
    out = {}
    threads = [
        threading.Thread(target=one, args=(url, model, i, max_tokens, out))
        for i in range(batch)
    ]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    ttft = [s[0] - t for (t, s) in out.values() if s]
    itl = [b - a for (_, s) in out.values() for a, b in zip(s, s[1:])]
    ntok = sum(len(s) for _, s in out.values())
    return {
        "batch": batch,
        "requests": len(out),
        "tokens": ntok,
        "wall_s": wall,
        "ttft_p50_ms": statistics.median(ttft) * 1000,
        "ttft_p95_ms": pct(ttft, 0.95) * 1000,
        "itl_p50_ms": statistics.median(itl) * 1000,
        "itl_p95_ms": pct(itl, 0.95) * 1000,
        "tok_per_s": ntok / wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--repetition-penalty", type=float)
    args = ap.parse_args()
    SAMPLING["temperature"] = args.temperature
    if args.top_k is not None:
        SAMPLING["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        SAMPLING["repetition_penalty"] = args.repetition_penalty

    # Warm-up: the first request after a server starts pays one-off costs
    # (CUDA context, graph capture) that are not steady-state latency.
    run_batch(args.url, args.model, 1, 8)

    rows = []
    print(f"| {args.label}: batch | TTFT p50 / p95 | ITL p50 / p95 | tokens/s |")
    print("|---|---|---|---|")
    for b in (int(x) for x in args.batches.split(",")):
        r = run_batch(args.url, args.model, b, args.max_tokens)
        rows.append(r)
        print(
            f"| {b} | {r['ttft_p50_ms']:.0f} / {r['ttft_p95_ms']:.0f} ms "
            f"| {r['itl_p50_ms']:.1f} / {r['itl_p95_ms']:.1f} ms | {r['tok_per_s']:.0f} |"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.label}.json").write_text(
        json.dumps({"label": args.label, "url": args.url, "model": args.model,
                    "max_tokens": args.max_tokens, "sampling": SAMPLING, "rows": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
