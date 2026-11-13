#!/usr/bin/env python3
"""Measure TTFT, ITL, TPOT and throughput against any OpenAI-compatible endpoint.

Standard library only. No SDK, no pandas, no plotting.

  python 04_speed_test.py --url http://localhost:8000 --requests 32 --concurrency 8
  python 04_speed_test.py --selftest

Token counts are estimated by counting non-empty delta.content fragments, so a
token split across SSE chunks undercounts. Good enough for latency; use the
server's own counters for billing.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def parse_sse(lines):
    """Return token fragments from an SSE stream. Keepalives and [DONE] are dropped."""
    frags = []
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        piece = (choices[0].get("delta") or {}).get("content")
        if piece:
            frags.append(piece)
    return frags


def measure_stream(chunks):
    """chunks: iterable of (timestamp, fragment). Returns latency metrics."""
    times = [t for t, frag in chunks if frag]
    if not times:
        return {"ttft_s": 0.0, "itl_s": [], "output_tokens": 0, "tpot_s": 0.0}
    ttft = times[0]
    itl = [b - a for a, b in zip(times, times[1:])]
    n = len(times)
    tpot = (times[-1] - times[0]) / (n - 1) if n > 1 else 0.0
    return {"ttft_s": ttft, "itl_s": itl, "output_tokens": n, "tpot_s": tpot}


def one_request(url, model, prompt, max_tokens, api_key):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key}"})
    t0 = time.perf_counter()
    chunks = []
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                for frag in parse_sse([line]):
                    chunks.append((time.perf_counter() - t0, frag))
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode("utf-8", "replace")[:500], file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        print(f"connection failed: {e.reason}", file=sys.stderr)
        return None
    m = measure_stream(chunks)
    m["e2e_s"] = time.perf_counter() - t0
    return m


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return s[k]


def summarize(results, wall_s):
    ok = [r for r in results if r]
    if not ok:
        return {"requests": 0}
    ttft = [r["ttft_s"] for r in ok]
    itl = [x for r in ok for x in r["itl_s"]]
    out_tokens = sum(r["output_tokens"] for r in ok)
    return {
        "requests": len(ok),
        "failed": len(results) - len(ok),
        "wall_s": round(wall_s, 3),
        "ttft": {f"p{p}": round(pct(ttft, p), 3) for p in (50, 95, 99)},
        "itl": {f"p{p}": round(pct(itl, p) * 1000, 1) for p in (50, 95, 99)},
        "tpot_ms": round(pct([r["tpot_s"] for r in ok], 50) * 1000, 1),
        "output_tok_per_s": round(out_tokens / wall_s, 1),
    }


def selftest():
    sse = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{}}]}',                     # keepalive, no content
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        ": ping",
        "data: [DONE]",
    ]
    frags = parse_sse(sse)
    ok1 = frags == ["Hel", "lo"]
    print(f"{'PASS' if ok1 else 'FAIL'} parse_sse: {frags}")

    chunks = [(0.250, "a")] + [(0.250 + 0.010 * i, "x") for i in range(1, 20)]
    m = measure_stream(chunks)
    ok2 = abs(m["ttft_s"] - 0.250) < 1e-6
    ok3 = abs(statistics.mean(m["itl_s"]) - 0.010) < 1e-6
    ok4 = abs(m["tpot_s"] - 0.010) < 1e-6  # (0.440 - 0.250) / 19 gaps, hand-computed
    print(f"{'PASS' if ok2 else 'FAIL'} ttft: {m['ttft_s']:.3f}")
    print(f"{'PASS' if ok3 else 'FAIL'} mean itl: {statistics.mean(m['itl_s']):.4f}")
    print(f"{'PASS' if ok4 else 'FAIL'} tpot: {m['tpot_s']:.4f}")
    return 0 if all([ok1, ok2, ok3, ok4]) else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="llama-3.1-8b-instruct")
    ap.add_argument("--api-key", default="sk-local-dev")
    ap.add_argument("--prompt", default="Explain what a KV cache is in three sentences.")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    a.concurrency = max(1, a.concurrency)  # ThreadPoolExecutor raises ValueError on 0

    if a.selftest:
        return selftest()

    print(f"{a.requests} requests, concurrency {a.concurrency}, max_tokens {a.max_tokens}")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        results = list(pool.map(
            lambda _: one_request(a.url, a.model, a.prompt, a.max_tokens, a.api_key),
            range(a.requests)))
    s = summarize(results, time.perf_counter() - t0)

    if not s["requests"]:
        print("no successful requests")
        return 1
    print(f"\nrequests {s['requests']} ok, {s['failed']} failed, wall {s['wall_s']}s")
    print(f"TTFT  p50 {s['ttft']['p50']}s  p95 {s['ttft']['p95']}s  p99 {s['ttft']['p99']}s")
    print(f"ITL   p50 {s['itl']['p50']}ms p95 {s['itl']['p95']}ms p99 {s['itl']['p99']}ms")
    print(f"TPOT  p50 {s['tpot_ms']}ms")
    print(f"output throughput {s['output_tok_per_s']} tok/s")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=1)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
