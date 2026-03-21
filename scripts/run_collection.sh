#!/bin/bash
# Run DeepSeek-V2-Lite inference with MoE routing trace collection via SGLang runtime
# across all prompts in data/prompt_suite.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

MODEL_PATH="${ROOT_DIR}/models/deepseek-ai/DeepSeek-V2-Lite-Chat"
AMX_PATH="${ROOT_DIR}/models/DeepSeek-V2-Lite-Chat-AMXINT4"
PROMPT_SUITE="${ROOT_DIR}/data/prompt_suite.json"
OUTPUT_DIR="${ROOT_DIR}/data/traces"
AGGREGATED_TRACE_FILE="${OUTPUT_DIR}/live_capture.parquet"
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
rm -f "${AGGREGATED_TRACE_FILE}" "${OUTPUT_DIR}"/*_session.parquet
rm -f "${OUTPUT_DIR}"/output_*.json "${OUTPUT_DIR}/capture_summary.json"

echo "=== MoE Routing Collection (kt-kernel + sglang) ==="
echo "Model: ${MODEL_PATH}"
echo "CPU Weights: ${AMX_PATH}"
echo "Prompt source: ${PROMPT_SUITE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Aggregated trace: ${AGGREGATED_TRACE_FILE}"

PROMPTS_JSON=$(python3 -c "
import json
from pathlib import Path
suite = json.loads(Path('${PROMPT_SUITE}').read_text())
print(json.dumps(suite['prompts']))
")

TOTAL_PROMPTS=$(python3 - "${PROMPTS_JSON}" <<'PY'
import json
import sys
print(len(json.loads(sys.argv[1])))
PY
)

echo "Total prompts to process: ${TOTAL_PROMPTS}"

export CUDA_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime

python3 - "${PROMPTS_JSON}" "${MODEL_PATH}" "${AMX_PATH}" "${OUTPUT_DIR}" "${PORT}" "${MAX_TOKENS}" <<'PYEOF'
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

prompts = json.loads(sys.argv[1])
model_path = sys.argv[2]
amx_path = sys.argv[3]
output_dir = Path(sys.argv[4])
port = sys.argv[5]
max_tokens = int(sys.argv[6])

results = []


def wait_healthy() -> bool:
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return True
        except Exception:
            time.sleep(2)
    return False


for i, prompt_entry in enumerate(prompts):
    prompt_id = prompt_entry["id"]
    category = prompt_entry["category"]
    prompt_text = prompt_entry["prompt"]
    context_id = f"ctx_{prompt_id}"
    trace_file = output_dir / f"{prompt_id}_session.parquet"

    print(f"[{i+1}/{len(prompts)}] Processing prompt: {prompt_id} (category: {category})", flush=True)

    env = os.environ.copy()
    env["KT_MOE_ROUTING_RECORD"] = "true"
    env["KT_MOE_ROUTING_TRACE_DIR"] = str(output_dir)
    env["KT_MOE_ROUTING_TRACE_FILE"] = str(trace_file)
    env["KT_MOE_ROUTING_PROMPT_ID"] = prompt_id
    env["KT_MOE_ROUTING_CONTEXT_ID"] = context_id
    env["KT_MOE_ROUTING_TOKEN_CATEGORY"] = category

    log_file = output_dir / f"collection_server_{prompt_id}.log"
    with open(log_file, "w", encoding="utf-8") as lf:
        server = subprocess.Popen(
            [
                "python3",
                "-m",
                "sglang.launch_server",
                "--host",
                "0.0.0.0",
                "--port",
                port,
                "--model",
                model_path,
                "--kt-weight-path",
                amx_path,
                "--kt-cpuinfer",
                "25",
                "--kt-threadpool-count",
                "1",
                "--kt-num-gpu-experts",
                "1",
                "--kt-method",
                "AMXINT4",
                "--kt-gpu-prefill-token-threshold",
                "4096",
                "--kt-enable-dynamic-expert-update",
                "--attention-backend",
                "triton",
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.90",
                "--chunked-prefill-size",
                "4096",
                "--max-running-requests",
                "32",
                "--max-total-tokens",
                "40000",
                "--watchdog-timeout",
                "3000",
                "--enable-mixed-chunk",
                "--tensor-parallel-size",
                "1",
                "--enable-p2p-check",
                "--disable-shared-experts-fusion",
            ],
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    try:
        if not wait_healthy():
            raise RuntimeError("server failed to become healthy")

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "DeepSeek-V2-Lite-Chat",
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            response = json.loads(r.read().decode("utf-8"))
            generated_text = response["choices"][0]["message"]["content"]

        output_file = output_dir / f"output_{prompt_id}.json"
        output_file.write_text(
            json.dumps(
                {
                    "prompt_id": prompt_id,
                    "category": category,
                    "context_id": context_id,
                    "prompt_text": prompt_text,
                    "generated_text": generated_text,
                    "timestamp": time.time(),
                    "trace_file": str(trace_file),
                    "server_log": str(log_file),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if not trace_file.exists():
            raise RuntimeError(f"trace file missing: {trace_file}")

        print(f"  ✓ Saved output to {output_file}", flush=True)
        results.append(
            {
                "prompt_id": prompt_id,
                "output_file": str(output_file),
                "trace_file": str(trace_file),
                "server_log": str(log_file),
                "success": True,
            }
        )
    except Exception as e:
        print(f"  ✗ Error processing {prompt_id}: {e}", flush=True)
        results.append(
            {
                "prompt_id": prompt_id,
                "trace_file": str(trace_file),
                "server_log": str(log_file),
                "error": str(e),
                "success": False,
            }
        )
    finally:
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGINT)
        except Exception:
            pass
        try:
            server.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            except Exception:
                pass
            server.wait(timeout=30)

    time.sleep(1)

summary_file = output_dir / "capture_summary.json"
summary_file.write_text(
    json.dumps(
        {
            "total": len(prompts),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "results": results,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"\nCapture summary saved to {summary_file}", flush=True)
PYEOF

echo ""
echo "=== Collection Complete ==="
echo "Output files saved to: ${OUTPUT_DIR}"

echo ""
echo "Aggregating traces..."
python3 - <<PY
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

traces_dir = Path('${OUTPUT_DIR}')
summary_file = traces_dir / 'capture_summary.json'
output_file = Path('${AGGREGATED_TRACE_FILE}')

if not summary_file.exists():
    raise SystemExit(f'Missing summary file: {summary_file}')

summary = json.loads(summary_file.read_text())
trace_files = [Path(r['trace_file']) for r in summary.get('results', []) if r.get('success')]
trace_files = [p for p in trace_files if p.exists()]

if not trace_files:
    raise SystemExit('No per-prompt trace files found to aggregate')

tables = [pq.read_table(p) for p in trace_files]
combined = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options='default')
pq.write_table(combined, output_file, compression='zstd')

print(f'Aggregated trace saved to: {output_file}')
print(f'Total records: {combined.num_rows}')
print(f'Source trace files: {len(trace_files)}')
PY

echo ""
echo "Next steps:"
echo "  - Analyze traces: python3 -m kt_kernel.moe_routing.analyze --trace-file ${AGGREGATED_TRACE_FILE} --output-dir ${ROOT_DIR}/data/analysis"
echo "  - View generated outputs: ls -la ${OUTPUT_DIR}/output_*.json"
