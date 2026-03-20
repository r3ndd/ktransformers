#!/bin/bash
# Run DeepSeek-V2-Lite inference with MoE routing trace collection via SGLang runtime

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

MODEL_PATH="${ROOT_DIR}/models/deepseek-ai/DeepSeek-V2-Lite-Chat"
AMX_PATH="${ROOT_DIR}/models/DeepSeek-V2-Lite-Chat-AMXINT4"
PROMPT_SUITE="${ROOT_DIR}/data/prompt_suite.json"
OUTPUT_DIR="${ROOT_DIR}/data/traces"
TRACE_FILE="${OUTPUT_DIR}/live_capture.parquet"
PORT="10093"
MAX_TOKENS="${MAX_TOKENS:-64}"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Error: model not found at ${MODEL_PATH}"
  exit 1
fi

if [ ! -d "${AMX_PATH}" ]; then
  echo "Error: AMX weights not found at ${AMX_PATH}"
  exit 1
fi

if [ ! -f "${PROMPT_SUITE}" ]; then
  echo "Error: prompt suite not found at ${PROMPT_SUITE}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
rm -f "${TRACE_FILE}"

PROMPT_TEXT="$(python3 - <<PY
import json
from pathlib import Path
suite = json.loads(Path('${PROMPT_SUITE}').read_text())
first = suite['prompts'][0]['prompt']
print(first)
PY
)"

echo "=== MoE Routing Collection (kt-kernel + sglang) ==="
echo "Model: ${MODEL_PATH}"
echo "CPU Weights: ${AMX_PATH}"
echo "Prompt source: ${PROMPT_SUITE} (first prompt)"
echo "Trace output: ${TRACE_FILE}"

export CUDA_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime
export KT_MOE_ROUTING_RECORD=true
export KT_MOE_ROUTING_TRACE_DIR="${OUTPUT_DIR}"
export KT_MOE_ROUTING_TRACE_FILE="${TRACE_FILE}"
export KT_MOE_ROUTING_PROMPT_ID="live_capture"
export KT_MOE_ROUTING_CONTEXT_ID="live_capture_ctx"
export KT_MOE_ROUTING_TOKEN_CATEGORY="assistant"

setsid python3 -m sglang.launch_server \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --kt-weight-path "${AMX_PATH}" \
  --kt-cpuinfer 25 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 1 \
  --kt-method AMXINT4 \
  --kt-gpu-prefill-token-threshold 4096 \
  --kt-enable-dynamic-expert-update \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 4096 \
  --max-running-requests 32 \
  --max-total-tokens 40000 \
  --watchdog-timeout 3000 \
  --enable-mixed-chunk \
  --tensor-parallel-size 1 \
  --enable-p2p-check \
  --disable-shared-experts-fusion \
  --rl-on-policy-target fsdp \
  > "${OUTPUT_DIR}/collection_server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    # Send SIGINT to the whole process group so scheduler/worker processes shutdown cleanly.
    kill -INT -- "-${SERVER_PID}" || true
    wait "${SERVER_PID}" || true
    sleep 2
  fi
}
trap cleanup EXIT

echo "Waiting for server health..."
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
  echo "Error: server failed to become healthy"
  exit 1
fi

python3 - <<PY
import json
import urllib.request

url = 'http://127.0.0.1:${PORT}/v1/chat/completions'
payload = {
  'model': 'DeepSeek-V2-Lite-Chat',
  'messages': [{'role': 'user', 'content': '''${PROMPT_TEXT}'''}],
  'max_tokens': int('${MAX_TOKENS}'),
  'temperature': 0.0,
}
req = urllib.request.Request(
  url,
  data=json.dumps(payload).encode('utf-8'),
  headers={'Content-Type': 'application/json'},
)
with urllib.request.urlopen(req, timeout=300) as r:
  print('Completion status:', r.status)
PY

cleanup
trap - EXIT

python3 - <<PY
from pathlib import Path
import pyarrow.parquet as pq
p = Path('${TRACE_FILE}')
if not p.exists():
  raise SystemExit(f'Missing trace file: {p}')
t = pq.read_table(p)
print(f'Trace rows: {t.num_rows}')
print(f'Trace file: {p}')
PY

echo "Collection complete."
echo "Next: python3 -m kt_kernel.moe_routing.analyze --trace-file ${TRACE_FILE} --output-dir ${ROOT_DIR}/data/analysis"
