#!/usr/bin/env bash
# Launch an OpenAI-compatible inference server with the flags that matter.
# Every flag below is explained in Chapter 5. Edit the config block, run it.
#
#   bash 03_launch_server.sh              # start and wait for health
#   bash 03_launch_server.sh --dry-run    # print the command, run nothing
#
# Note: vLLM's --api-key guards /v1, /v2 and /inference, NOT /invocations.
# Put a real gateway in front of anything public.
set -euo pipefail

# ---- config ---------------------------------------------------------------
ENGINE="${ENGINE:-vllm}"                       # vllm | sglang
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"                           # sglang defaults to 30000
API_KEY="${API_KEY:-sk-local-dev}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_SEQS="${MAX_SEQS:-64}"
KV_DTYPE="${KV_DTYPE:-auto}"                   # auto | fp8 (sglang: mapped to fp8_e4m3)
TP="${TP:-1}"

# vLLM accepts bare "fp8"; sglang v0.5.23 wants fp8_e4m3/fp8_e5m2. Map it for sglang.
if [ "$ENGINE" = "sglang" ] && [ "$KV_DTYPE" = "fp8" ]; then
  KV_DTYPE="fp8_e4m3"
fi

if [ "$ENGINE" = "sglang" ]; then
  CMD=(sglang serve "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --mem-fraction-static "$GPU_UTIL"          # vLLM's --gpu-memory-utilization
    --context-length "$MAX_LEN"
    --max-running-requests "$MAX_SEQS"
    --tp-size "$TP"
    --kv-cache-dtype "$KV_DTYPE"
    --api-key "$API_KEY")
else
  CMD=(vllm serve "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --served-model-name "${MODEL##*/}"         # stable public model id
    --max-model-len "$MAX_LEN"                 # hard cap: exceed it and prefill OOMs
    --gpu-memory-utilization "$GPU_UTIL"       # weights + KV; above 0.95 invites OOM kills
    --max-num-seqs "$MAX_SEQS"                 # concurrency cap; raise only if KV allows
    --kv-cache-dtype "$KV_DTYPE"               # fp8 halves KV, needs calibration for accuracy
    --enable-prefix-caching                    # default on since V1; explicit for readers
    --tensor-parallel-size "$TP"
    --api-key "$API_KEY")
fi

if [ "${1:-}" = "--dry-run" ]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

echo "starting ${ENGINE} on :${PORT} ..."
"${CMD[@]}" &
SERVER_PID=$!

for i in $(seq 1 180); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "ready: health endpoint responds. It does not prove generation works."
    cat <<EOF

test it:
  curl -s http://localhost:${PORT}/v1/chat/completions \\
    -H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json" \\
    -d '{"model":"${MODEL##*/}","messages":[{"role":"user","content":"Say hi in five words."}],"max_tokens":32}'

docker equivalent (thin wrapper, see Appendix C):
  docker run --gpus all -p ${PORT}:8000 -v ~/.cache/huggingface:/root/.cache/huggingface vllm/vllm-openai:v0.33.0 ${MODEL} --max-model-len ${MAX_LEN}
EOF
    wait "$SERVER_PID"
    exit 0
  fi
  sleep 1
done

echo "server did not become healthy in 180s. First launch downloads weights; a 16 GB model"
echo "at 0.3-0.5 GB/s takes 30-60s, and CUDA graph capture adds 10-60s on top."
kill "$SERVER_PID" 2>/dev/null || true
exit 1
