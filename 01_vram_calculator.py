#!/usr/bin/env python3
"""VRAM and concurrency calculator for LLM inference.

Standard library only. No GPU, no AI packages, runs on a laptop.

Implements the book's formulas F-1 (weights), F-2 (KV per token), F-3 (total),
F-4 (max sequences), F-7 (ridge point) and F-8 (decode ceiling).

Units: GB = 10^9 bytes, GiB = 2^30 bytes. Card capacity is quoted in GiB.
AWQ/GPTQ weights are ~4.25 bits, GGUF Q4_K_M ~4.85, so --bits 4 understates
real quantized size by 6-20%. Budget accordingly.
"""
from __future__ import annotations

import argparse
import json
import sys

GIB = 2 ** 30
H100_BW = 3350.0  # GB/s, H100 SXM, the reference card for F-8

# name: (params, n_layers, n_kv_heads, head_dim) -- read from each model's config.json
PRESETS = {
    "llama31-8b": (8.03e9, 32, 8, 128),
    "llama31-70b": (70.6e9, 80, 8, 128),
    "qwen25-7b": (7.62e9, 28, 4, 128),
    "qwen3-8b": (8.19e9, 36, 8, 128),
    "mistral-7b": (7.24e9, 32, 8, 128),
}
CARDS = (24, 48, 80, 141, 192, 288)


def elem_bytes(bits: int) -> float:
    return bits / 8


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, kv_elem_bytes: float) -> int:
    """F-2. Two tensors (K and V) per layer, per token."""
    return int(2 * layers * kv_heads * head_dim * kv_elem_bytes)


def weights_gib(params: float, bits: float) -> float:
    """F-1."""
    return params * bits / 8 / GIB


def workspace_gib(card_gib: float) -> float:
    """CUDA context, CUDA graphs and the logits buffer. 8% of card, clamped [2, 6] GiB."""
    return min(max(2.0, 0.08 * card_gib), 6.0)


def total_gib(params, bits, ctx, layers, kv_heads, head_dim, kv_elem_bytes, card_gib) -> float:
    """F-3."""
    return weights_gib(params, bits) + kv_bytes_per_token(layers, kv_heads, head_dim, kv_elem_bytes) * ctx / GIB + workspace_gib(card_gib)


def max_seqs(card_gib, params, bits, layers, kv_heads, head_dim, kv_elem_bytes, ctx, gpus: int = 1) -> int:
    """F-4. Uses 90% of the card, matching vLLM's --gpu-memory-utilization default intent.
    TP-aware: with gpus > 1, weights and KV split across the group; the workspace does not."""
    budget = 0.9 * card_gib - weights_gib(params, bits) / gpus - workspace_gib(card_gib)
    per_seq = kv_bytes_per_token(layers, kv_heads, head_dim, kv_elem_bytes) * ctx / GIB / gpus
    if budget <= 0 or per_seq <= 0:
        return 0
    return max(0, int(budget / per_seq))


def ceiling_tok_s(bw_gbs, params, bits, ctx, layers, kv_heads, head_dim, kv_elem_bytes) -> float:
    """F-8. Single-stream decode ceiling: bandwidth / bytes read per token."""
    per_token = params * bits / 8 + kv_bytes_per_token(layers, kv_heads, head_dim, kv_elem_bytes) * ctx
    return bw_gbs * 1e9 / per_token


def ridge_point(tflops_dense: float, bw_gbs: float) -> float:
    """F-7. FLOP/byte. H100 SXM: 989e12 / 3.35e12 = 295.2."""
    return tflops_dense * 1e12 / (bw_gbs * 1e9)


def fmt(n: int) -> str:
    return f"{n:,}"


def report(preset, bits, kv_bits, card, gpus=1) -> list[str]:
    params, layers, kvh, d = PRESETS[preset]
    kvb = elem_bytes(kv_bits)
    w = weights_gib(params, bits) / gpus
    ws = workspace_gib(card)
    out = [
        f"model      : {preset}  ({params / 1e9:.2f}B params, {layers} layers, {kvh} KV heads, d={d})",
        f"weights    : {bits}-bit -> {w:.2f} GiB/GPU" + (f"  (TP={gpus})" if gpus > 1 else ""),
        f"KV         : {kv_bits}-bit, {kv_bytes_per_token(layers, kvh, d, kvb) / 1024:.0f} KiB/token",
        f"card       : {card} GiB, usable {0.9 * card:.1f} GiB, workspace {ws:.2f} GiB",
        "",
        f"{'ctx':>9} {'KV GiB':>8} {'total':>8} {'fits':>6} {'seqs':>6}",
    ]
    verdicts = []
    for ctx in (4096, 32768, 131072):
        kv = kv_bytes_per_token(layers, kvh, d, kvb) * ctx / GIB / gpus
        total = w + kv + ws
        fits = total <= 0.9 * card
        seqs = max_seqs(card, params, bits, layers, kvh, d, kvb, ctx, gpus)
        out.append(f"{fmt(ctx):>9} {kv:>8.2f} {total:>8.2f} {'yes' if fits else 'no':>6} {seqs:>6}")
        verdicts.append((ctx, fits, seqs))
    out.append("")
    if verdicts[0][1] and not verdicts[1][1]:
        out.append(f"verdict    : fits one {card} GiB card at 4k only.")
    elif verdicts[1][1] and not verdicts[2][1]:
        out.append(f"verdict    : fits one {card} GiB card up to 32k, not 128k.")
    elif verdicts[2][1]:
        out.append(f"verdict    : fits one {card} GiB card at 128k.")
    else:
        out.append(f"verdict    : does not fit {card} GiB at 4k. Lower --bits or raise --gpus (TP), then re-run.")
    out.append(f"notice     : 1M-token context is a different calculation; run --ctx 1000000 to see why nobody serves it.")
    return out


def table_ch4(as_json: bool = False) -> str:
    """The exact table printed in Chapter 4. Single source of truth for that table."""
    rows = []
    for preset in PRESETS:
        p, l, kh, d = PRESETS[preset]
        for bits in (16, 8, 4):
            for kvb in (2, 1):
                for ctx in (4096, 32768, 131072):
                    rows.append({
                        "model": preset, "bits": bits, "kv_bits": kvb * 8, "ctx": ctx,
                        "seqs": {c: max_seqs(c, p, bits, l, kh, d, kvb, ctx) for c in (24, 48, 80)},
                    })
    if as_json:
        return json.dumps(rows, indent=1)
    head = f"{'model':<12}{'W':>3}{'KV':>4}{'ctx':>9}{'24G':>7}{'48G':>7}{'80G':>7}"
    lines = ["Concurrency (max concurrent sequences at 90% VRAM)", head, "-" * len(head)]
    for r in rows:
        s = r["seqs"]
        lines.append(f"{r['model']:<12}{r['bits']:>3}{r['kv_bits']:>4}{fmt(r['ctx']):>9}{s[24]:>7}{s[48]:>7}{s[80]:>7}")
    return "\n".join(lines)


CASES = [
    ("kv_8b_fp16",  lambda: kv_bytes_per_token(32, 8, 128, 2), 131072, 0),
    ("w_8b_fp16",   lambda: weights_gib(8.03e9, 16), 14.96, 0.02),
    ("w_70b_fp8",   lambda: weights_gib(70.6e9, 8), 65.75, 0.05),
    ("total_8b_32k", lambda: total_gib(8.03e9, 16, 32768, 32, 8, 128, 1, 24), 18.96, 0.03),
    ("free_24g",    lambda: 24 - total_gib(8.03e9, 16, 32768, 32, 8, 128, 1, 24), 5.04, 0.03),
    ("seqs_24g_4k", lambda: max_seqs(24, 8.03e9, 16, 32, 8, 128, 1, 4096), 18, 0),
    ("seqs_80g_1m", lambda: max_seqs(80, 8.03e9, 4, 32, 8, 128, 1, 1_000_000), 1, 0),
    ("ws_80",       lambda: workspace_gib(80), 6.0, 0.0),
    ("ws_24",       lambda: workspace_gib(24), 2.0, 0.0),
    ("ridge",       lambda: ridge_point(989, 3350), 295.2, 0.1),
    ("ceiling_8b_4k", lambda: ceiling_tok_s(H100_BW, 8.03e9, 16, 4096, 32, 8, 128, 2), 201.8, 0.5),
    ("seqs_80g_tp2_32k", lambda: max_seqs(80, 70.6e9, 8, 80, 8, 128, 1, 32768, 2), 13, 0),
]


def selftest() -> int:
    failed = 0
    for label, fn, expected, tol in CASES:
        got = fn()
        ok = abs(got - expected) <= tol if tol else got == expected
        print(f"{'PASS' if ok else 'FAIL'} {label}: got {got} expected {expected} (+/-{tol})")
        failed += 0 if ok else 1
    print(f"{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VRAM and concurrency calculator")
    ap.add_argument("--model", choices=sorted(PRESETS), default="llama31-8b")
    ap.add_argument("--bits", type=int, default=16, choices=[32, 16, 8, 4])
    ap.add_argument("--kv-bits", type=int, default=16, choices=[16, 8])
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--vram", type=float, default=24)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--table", choices=["ch4"])
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    a.gpus = max(1, a.gpus)  # --gpus 0 would divide by zero below

    if a.selftest:
        return selftest()
    if a.table == "ch4":
        text = table_ch4(as_json=bool(a.json))
        if a.json:
            open(a.json, "w", encoding="utf-8").write(text)
            print(f"wrote {a.json}")
        else:
            print(text)
        return 0
    if a.ctx:
        params, layers, kvh, d = PRESETS[a.model]
        kvb = elem_bytes(a.kv_bits)
        w = weights_gib(params, a.bits) / a.gpus
        kv = kv_bytes_per_token(layers, kvh, d, kvb) * a.ctx / GIB / a.gpus
        total = w + kv + workspace_gib(a.vram)
        free = a.vram - total
        print(f"total {total:.2f} GiB, free {free:.2f} GiB, max seqs {max_seqs(a.vram, params, a.bits, layers, kvh, d, kvb, a.ctx, a.gpus)}")
        return 0
    print("\n".join(report(a.model, a.bits, a.kv_bits, a.vram, a.gpus)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
