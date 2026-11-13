#!/usr/bin/env python3
"""Quantize an 8B model for vLLM. AWQ (W4A16), FP8 (W8A8), or GGUF for llama.cpp.

IMPORTANT: run this in a SEPARATE virtualenv from vLLM.
  vLLM 0.33.0          pins compressed-tensors == 0.17.0
  llm-compressor 0.15.0 pins compressed-tensors == 0.20.1
Installing both in one environment produces a version conflict, not a faster model.

Usage:
  python 02_quantize_model.py --method fp8 --model meta-llama/Llama-3.1-8B-Instruct --out ./out --dry-run
  python 02_quantize_model.py --method awq --model meta-llama/Llama-3.1-8B-Instruct --out ./out
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

PIP = {
    "awq": 'uv pip install "llmcompressor==0.15.0" "compressed-tensors==0.20.1" transformers accelerate',
    "fp8": 'uv pip install "llmcompressor==0.15.0" "compressed-tensors==0.20.1" transformers accelerate',
    "gguf": "git clone --depth 1 https://github.com/ggml-org/llama.cpp && cmake -B llama.cpp/build && cmake --build llama.cpp/build -j",
}
# Observed on-disk sizes for Llama-3.1-8B output dirs. They carry config, tokenizer and
# quant overhead, so they sit above the book's F-1 weights-only arithmetic (AWQ: 3.97 GiB).
EXPECTED = {"awq": "~5.7 GB", "fp8": "~8.1 GB", "gguf": "~4.9 GB"}
MIN_SM = {"awq": 8.0, "fp8": 8.9, "gguf": 0.0}  # full compute capability; FP8 = Ada/Hopper/Blackwell


def dir_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


def build_recipe(method: str):
    """The 12-line core pasted into Chapter 3."""
    from llmcompressor.modifiers.quantization import QuantizationModifier
    if method == "awq":
        from llmcompressor.modifiers.transform.awq import AWQModifier
        return [AWQModifier(duo_scaling="both"),
                QuantizationModifier(targets="Linear", scheme="W4A16_ASYM", ignore=["lm_head"])]
    return QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])


def check_capability(method: str) -> None:
    import torch
    if method == "gguf" or not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    capability = major + minor / 10
    if capability < MIN_SM[method]:
        need = {"awq": "SM 8.0+ (Ampere or newer)", "fp8": "SM 8.9+ (Ada/Hopper/Blackwell)"}[method]
        print(f"{method.upper()} needs {need}. This GPU has compute capability {capability}.")
        if method == "fp8":
            print("On an A6000 (8.6) or A100 (8.0), FP8 W8A8 kernels do not exist. Use --method awq.")
        sys.exit(1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["awq", "fp8", "gguf"], default="fp8")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    out = a.out or (a.model.split("/")[-1] + "-" + a.method.upper())
    if a.dry_run:
        print(f"method      : {a.method}")
        print(f"model       : {a.model}")
        print(f"output      : {out}")
        print(f"separate venv required: yes (compressed-tensors pin conflict with vLLM)")
        print(f"install     : {PIP[a.method]}")
        print(f"expected size: {EXPECTED[a.method]}")
        if a.method == "gguf":
            print("steps       : convert_hf_to_gguf.py --outtype bf16, then llama-quantize Q4_K_M")
        else:
            print("recipe      : QuantizationModifier(targets='Linear', scheme=%s, ignore=['lm_head'])" % ("W4A16_ASYM" if a.method == "awq" else "FP8_DYNAMIC"))
        return 0

    if a.method == "gguf":
        print("Run the commands printed by --dry-run, then: llama-server -m <file>.gguf")
        return 0

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from llmcompressor import oneshot
    except ImportError:
        print("Missing packages. Install with:\n  " + PIP[a.method])
        return 1

    check_capability(a.method)
    print(f"loading {a.model} ...")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype="auto")
    tok = AutoTokenizer.from_pretrained(a.model)

    before = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    oneshot(model=model, recipe=build_recipe(a.method),
            dataset="perfectblend", splits="train[:512]", max_seq_length=512,
            num_calibration_samples=256)
    model.save_pretrained(out, save_compressed=True)
    tok.save_pretrained(out)

    after = dir_gb(out)
    print(f"in-memory before : {before:.2f} GB")
    print(f"on-disk after    : {after:.2f} GB")
    print(f"to serve         : vllm serve {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
