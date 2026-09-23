#!/usr/bin/env python3
"""End-to-end latency and throughput for /v1/decisions.

Measures what a caller sees, through HTTP and NATS, not the forward pass on
its own — `cargo run --example decision_bench` does that. Both are worth
having: the gap between them is the transport, and it is most of the cost of
a single short question.

    python benchmark/bench_decisions.py --url http://127.0.0.1:8080

The published reference figures, for scale: 39.5 ms for one question and
158.6 ms for ten on a Tesla T4, and 103-332 questions/sec batched.
"""

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request

STATE = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate "
    "today or we will cancel our plan.",
}

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}


def questions(n: int) -> dict:
    """`n` questions: the four real ones, then copies of the first."""
    out = dict(list(QUESTIONS.items())[:n])
    for i in range(len(out), n):
        out[f"extra_{i}"] = QUESTIONS["department"]
    return out


def call(url: str, n: int) -> float:
    body = json.dumps({"state": STATE, "questions": questions(n)}).encode()
    req = urllib.request.Request(url, body, {"content-type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        payload = json.load(r)
    if len(payload["answers"]) != n:
        raise SystemExit(f"asked {n} questions, got {len(payload['answers'])}")
    return (time.perf_counter() - start) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--iters", type=int, default=15)
    args = ap.parse_args()
    url = args.url.rstrip("/") + "/v1/decisions"

    print("one request at a time")
    print(f"{'questions':>10} {'p50 ms':>9} {'p95 ms':>9} {'ms/question':>13}")
    for n in (1, 4, 16):
        for _ in range(3):
            call(url, n)
        lat = sorted(call(url, n) for _ in range(args.iters))
        p50 = statistics.median(lat)
        p95 = lat[min(int(len(lat) * 0.95), len(lat) - 1)]
        print(f"{n:>10} {p50:>9.1f} {p95:>9.1f} {p50 / n:>13.2f}")

    print("\nconcurrent, four questions each")
    print(f"{'clients':>10} {'p50 ms':>9} {'req/s':>9} {'questions/s':>13}")
    for clients in (1, 8, 32):
        with concurrent.futures.ThreadPoolExecutor(clients) as pool:
            list(pool.map(lambda _: call(url, 4), range(clients)))
            start = time.perf_counter()
            lat = list(pool.map(lambda _: call(url, 4), range(clients * 4)))
            wall = time.perf_counter() - start
        n = clients * 4
        print(
            f"{clients:>10} {statistics.median(lat):>9.1f} "
            f"{n / wall:>9.1f} {n * 4 / wall:>13.1f}"
        )


if __name__ == "__main__":
    main()
