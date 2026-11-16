# Inference Engineering Handbook — companion scripts

Four files that back the book. You need a GPU only for file two.
**Get the Complete Book:** These scripts are the runnable companion code for ** https://www.lulu.com/shop/vedansh-dixit/step-by-step-inference-engineering/ebook/product-w46yr9g.html?page=1&pageSize=4 **—a practical guide covering KV-cache math, quantization trade-offs, continuous batching, and cloud unit economics.
## Setup in three steps

```bash
git clone <repo-url>                                    # or unzip it
cd inference-engineering-handbook
python 01_vram_calculator.py --selftest                # expect: 12/12 passed
```

Then read `01_vram_calculator.py` alongside Chapter 1 — it implements the chapter's formulas (F-1 to F-4, F-7, F-8) as runnable arithmetic.
## Why This Repo Exists

Moving an LLM into production requires answering core engineering questions regarding VRAM limits, quantization without breaking dependencies, and measuring true tail latency (TTFT, ITL, TPOT, p99). Running the full GPU pipeline on an on-demand NVIDIA L4 costs under \$2 total.


## What each file does

| File | What it is | Chapter |
|---|---|---|
| `01_vram_calculator.py` | Pure arithmetic. Does a model fit, and how many users fit with it. `--table ch4` prints the concurrency table. | 1, 4 |
| `02_quantize_model.py` | Shrinks an 8B model to AWQ 4-bit, FP8, or GGUF. `--dry-run` prints the recipe without touching a GPU. | 3 |
| `03_launch_server.sh` | One command for a vLLM or SGLang OpenAI-compatible server, with every flag commented. | 5 |
| `04_speed_test.py` | Streams requests and prints TTFT, ITL, TPOT and p99. | 2, 5, 6 |

## What you need

- File 01: Python 3.9+. No packages. Runs on a laptop.
- File 02: a GPU, plus a **separate virtualenv** — vLLM pins `compressed-tensors==0.17.0` and llm-compressor pins `==0.20.1`, so they cannot share one. The script prints the exact pip line for your method.
- File 03: a GPU host on Linux or WSL2. It is bash; it will not run in PowerShell.
- File 04: Python on any OS. Point it at any OpenAI-compatible endpoint.
- Cost of one pass of files 02-04 on an L4 ($0.80/hr on-demand): under **$2**.

## Expected output

```
$ python 01_vram_calculator.py --model llama31-8b --bits 16 --kv-bits 8 --ctx 32768 --vram 24
total 18.96 GiB, free 5.04 GiB, max seqs 2

$ python 04_speed_test.py --selftest
PASS parse_sse: ['Hel', 'lo']
PASS ttft: 0.250
PASS mean itl: 0.0100
PASS tpot: 0.0100
```

Code license: MIT (see LICENSE). The book itself: all rights reserved.

Grab the copy of this book @  https://www.lulu.com/shop/vedansh-dixit/step-by-step-inference-engineering/ebook/product-w46yr9g.html?page=1&pageSize=4 
