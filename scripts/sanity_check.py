#!/usr/bin/env python3
"""Sanity check: run one prompt through SGLang + kt-kernel and verify output quality."""

import http.client
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent

MODEL_PATH = str(
    Path(
        os.environ.get(
            "SANITY_MODEL_PATH",
            str(ROOT / "models/Qwen3.5-35B-A3B"),
        )
    ).resolve()
)
WEIGHT_PATH = Path(
    os.environ.get(
        "SANITY_WEIGHT_PATH",
        str(ROOT / "models/Qwen3.5-35B-A3B-GGUF-Q4_K_M"),
    )
)
SERVED_MODEL_NAME = os.environ.get("SANITY_MODEL_NAME", "Qwen3.5-35B-A3B")
KT_METHOD = os.environ.get("SANITY_KT_METHOD", "LLAMAFILE")
OUTPUT_DIR = ROOT / "data/traces"
DEFAULT_PORT = int(os.environ.get("SANITY_PORT", "10093"))
REQUIRE_TRACE = os.environ.get("SANITY_REQUIRE_TRACE", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
GEN_MAX_TOKENS = int(os.environ.get("SANITY_MAX_TOKENS", "96"))
GEN_TEMPERATURE = float(os.environ.get("SANITY_TEMPERATURE", "0.2"))
GEN_TOP_P = float(os.environ.get("SANITY_TOP_P", "0.9"))
PYTHON_BIN = str(ROOT / ".venv/bin/python")

PROMPTS = [
    {
        "id": "coding_001",
        "category": "coding",
        "prompt": "Write a Python function to find the longest palindromic substring in a given string. Include a brief explanation of your approach and time complexity.",
    },
    {
        "id": "reasoning_001",
        "category": "reasoning",
        "prompt": "A train travels from city A to city B at 60 mph and returns at 40 mph. If the total round trip takes 5 hours, how far apart are the two cities? Show your work step by step.",
    },
]


def _read_log_tail(log_file: Path, max_lines: int = 80) -> str:
    if not log_file.exists():
        return "<log file not created yet>"
    lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])


def _fatal_log_reason(log_file: Path):
    if not log_file.exists():
        return None
    text = log_file.read_text(encoding="utf-8", errors="ignore")
    fatal_markers = [
        "Received sigquit from a child process",
        "Scheduler hit an exception",
        "Fatal Python error",
        "triton.compiler.errors.CompilationError",
        "RuntimeError:",
        # FlashInfer/CUDA sampling dependency errors
        "curand.h: No such file or directory",
        "cannot find -lcurand",
        "curand.lib: cannot open",
        "flashinfer.*sampling.*compile",
        "flashinfer.*curand",
        "missing curand",
    ]
    for marker in fatal_markers:
        if marker in text:
            return marker
    return None


def _is_connection_error(err: Exception) -> bool:
    """Check if error is a connection-level failure (not startup-not-ready)."""
    if isinstance(err, http.client.RemoteDisconnected):
        return True
    if isinstance(err, ConnectionError):
        return True
    err_str = str(err).lower()
    fatal_patterns = [
        "connection reset",
        "connection closed",
        "broken pipe",
        "remote end closed",
    ]
    return any(p in err_str for p in fatal_patterns)


def wait_healthy(port, server_proc, log_file: Path, timeout=180):
    """Wait for server health, distinguishing startup-not-ready from fatal connection drops."""
    consecutive_conn_errors = 0
    max_conn_errors = 3  # Treat repeated connection errors as fatal

    for _ in range(timeout):
        if server_proc.poll() is not None:
            raise RuntimeError(
                f"Server process exited early with code {server_proc.returncode}. "
                f"Log tail:\n{_read_log_tail(log_file)}"
            )

        fatal_reason = _fatal_log_reason(log_file)
        if fatal_reason:
            raise RuntimeError(
                f"Detected fatal server error before health check ({fatal_reason}). "
                f"Log tail:\n{_read_log_tail(log_file)}"
            )

        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return True
        except urllib.error.HTTPError as e:
            # HTTP errors (5xx, etc.) are startup-not-ready, not fatal
            consecutive_conn_errors = 0
            time.sleep(1)
        except Exception as e:
            # Check if this is a connection-level error that suggests fatal issue
            if _is_connection_error(e):
                consecutive_conn_errors += 1
                if consecutive_conn_errors >= max_conn_errors:
                    raise RuntimeError(
                        f"Server connection dropped repeatedly ({consecutive_conn_errors} times). "
                        f"Likely fatal startup failure. Log tail:\n{_read_log_tail(log_file)}"
                    ) from e
            else:
                consecutive_conn_errors = 0
            time.sleep(1)

    raise RuntimeError(
        f"Server failed health check within {timeout}s. Log tail:\n{_read_log_tail(log_file)}"
    )


def _pick_available_port(preferred_port: int) -> int:
    """Use preferred port if free; otherwise ask OS for a free one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred_port))
            return preferred_port
        except OSError:
            pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_prompt(prompt_entry, fail_fast=False):
    prompt_id = prompt_entry["id"]
    category = prompt_entry["category"]
    prompt_text = prompt_entry["prompt"]
    context_id = f"ctx_{prompt_id}"
    trace_file = OUTPUT_DIR / f"{prompt_id}_session.parquet"
    log_file = OUTPUT_DIR / f"collection_server_{prompt_id}.log"

    for p in [trace_file, OUTPUT_DIR / f"output_{prompt_id}.json"]:
        if p.exists():
            p.unlink()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path_obj = Path(MODEL_PATH)
    port = _pick_available_port(DEFAULT_PORT)
    if port != DEFAULT_PORT:
        print(
            f"[sanity_check] Port {DEFAULT_PORT} is in use; falling back to {port} for {prompt_id}.",
            flush=True,
        )

    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"Weight path not found: {WEIGHT_PATH}")

    if not model_path_obj.exists():
        raise FileNotFoundError(
            "Local SANITY_MODEL_PATH does not exist. "
            f"Expected local model directory at: {model_path_obj}. "
            "Download with: hf download Qwen/Qwen3.5-35B-A3B --local-dir "
            f"{model_path_obj}"
        )

    has_index = (model_path_obj / "model.safetensors.index.json").exists()
    has_shard = any(model_path_obj.glob("*.safetensors"))
    if not (has_index and has_shard):
        raise FileNotFoundError(
            "Local SANITY_MODEL_PATH is missing safetensors checkpoint files. "
            f"Expected model.safetensors.index.json and *.safetensors in: {model_path_obj}"
        )

    env = os.environ.copy()
    env["KT_MOE_ROUTING_RECORD"] = "true"
    env["KT_MOE_ROUTING_TRACE_DIR"] = str(OUTPUT_DIR)
    env["KT_MOE_ROUTING_TRACE_FILE"] = str(trace_file)
    env["KT_MOE_ROUTING_PROMPT_ID"] = prompt_id
    env["KT_MOE_ROUTING_CONTEXT_ID"] = context_id
    env["KT_MOE_ROUTING_TOKEN_CATEGORY"] = category
    cuda_home = Path(
        os.environ.get(
            "SANITY_CUDA_HOME",
            "/usr/local/cuda-13.2"
            if Path("/usr/local/cuda-13.2").exists()
            else str(ROOT / ".venv/lib/python3.12/site-packages/nvidia/cuda_runtime"),
        )
    )
    cuda_header_candidates = [
        cuda_home / "include/cuda_runtime.h",
        cuda_home / "targets/x86_64-linux/include/cuda_runtime.h",
    ]
    if not any(p.exists() for p in cuda_header_candidates):
        raise FileNotFoundError(
            "CUDA toolkit headers not found (missing cuda_runtime.h). "
            f"Checked: {cuda_header_candidates}. "
            "Install CUDA runtime development headers (Fedora: cuda-cudart-devel-13-2)."
        )

    # Preflight check for curand headers/libraries (required for FlashInfer sampling)
    curand_header_candidates = [
        cuda_home / "include/curand.h",
        cuda_home / "targets/x86_64-linux/include/curand.h",
    ]
    curand_lib_candidates = [
        cuda_home / "lib64/libcurand.so",
        cuda_home / "targets/x86_64-linux/lib/libcurand.so",
        cuda_home / "lib/libcurand.so",
    ]
    if not any(p.exists() for p in curand_header_candidates):
        raise FileNotFoundError(
            "CUDA curand headers not found (missing curand.h). "
            f"Checked: {curand_header_candidates}. "
            "Install CUDA curand development package (Fedora: cuda-curand-devel-13-2, Ubuntu: cuda-curand-dev-13-2)."
        )
    if not any(p.exists() for p in curand_lib_candidates):
        raise FileNotFoundError(
            "CUDA curand library not found (missing libcurand.so). "
            f"Checked: {curand_lib_candidates}. "
            "Install CUDA curand runtime library (Fedora: cuda-curand-13-2, Ubuntu: cuda-curand-13-2)."
        )

    env["CUDA_HOME"] = str(cuda_home)
    env.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
    env.setdefault("KT_KERNEL_CPU_VARIANT", "avx2")
    env.setdefault("TVM_FFI_GPU_BACKEND", "cuda")
    env["PATH"] = f"{ROOT / '.venv' / 'bin'}:{cuda_home / 'bin'}:{env.get('PATH', '')}"

    with open(log_file, "w", encoding="utf-8") as lf:
        server = subprocess.Popen(
            [
                PYTHON_BIN,
                "-m",
                "sglang.launch_server",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
                "--model",
                str(MODEL_PATH),
                "--kt-weight-path",
                str(WEIGHT_PATH),
                "--kt-cpuinfer",
                "25",
                "--kt-threadpool-count",
                "1",
                "--kt-num-gpu-experts",
                "1",
                "--kt-method",
                KT_METHOD,
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

    result_payload = None

    try:
        wait_healthy(port, server, log_file)

        # Post-health liveness gate: verify server is still alive before request
        if server.poll() is not None:
            raise RuntimeError(
                f"Server process exited after health check (code {server.returncode}). "
                f"Log tail:\n{_read_log_tail(log_file)}"
            )

        fatal_reason = _fatal_log_reason(log_file)
        if fatal_reason:
            raise RuntimeError(
                f"Detected fatal server error after health check ({fatal_reason}). "
                f"Log tail:\n{_read_log_tail(log_file)}"
            )

        # Short stabilization delay to ensure server is ready for requests
        time.sleep(0.5)

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": SERVED_MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": GEN_MAX_TOKENS,
                    "temperature": GEN_TEMPERATURE,
                    "top_p": GEN_TOP_P,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        # Wrap request to fail fast on connection errors with log context
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                response = json.loads(r.read().decode("utf-8"))
        except (http.client.RemoteDisconnected, ConnectionError) as conn_err:
            # Fail fast on connection-level errors - likely server died
            raise RuntimeError(
                f"Connection lost during request (server likely died). "
                f"Error: {conn_err}. Log tail:\n{_read_log_tail(log_file)}"
            ) from conn_err
        except urllib.error.HTTPError as http_err:
            # HTTP errors may be transient - check if server is still alive
            if server.poll() is not None:
                raise RuntimeError(
                    f"HTTP error {http_err.code} and server process has exited. "
                    f"Log tail:\n{_read_log_tail(log_file)}"
                ) from http_err
            raise
        generated_text = response["choices"][0]["message"]["content"]

        output_file = OUTPUT_DIR / f"output_{prompt_id}.json"
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

        result_payload = {
            "prompt_id": prompt_id,
            "success": True,
            "generated_text": generated_text,
        }

    except Exception as e:
        if fail_fast:
            raise RuntimeError(
                f"Sanity check failed for {prompt_id}: {e}. See log: {log_file}"
            ) from e
        return {"prompt_id": prompt_id, "success": False, "error": str(e)}

    finally:
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            server.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            except Exception:
                pass
            server.wait(timeout=30)

        time.sleep(1)

    # Trace file may only be finalized during server shutdown; verify after cleanup.
    if result_payload is not None:
        for _ in range(10):
            if trace_file.exists():
                break
            time.sleep(0.5)
        if not trace_file.exists():
            if REQUIRE_TRACE:
                raise RuntimeError(
                    f"trace file missing after server shutdown: {trace_file}. "
                    f"Log tail:\n{_read_log_tail(log_file)}"
                )
            print(
                f"[sanity_check] WARNING: trace file missing for {prompt_id} ({trace_file}); continuing.",
                flush=True,
            )
        return result_payload


if __name__ == "__main__":
    fail_fast = os.environ.get("SANITY_FAIL_FAST", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    results = []
    for prompt_entry in PROMPTS:
        print(f"Running {prompt_entry['id']}...", flush=True)
        result = run_prompt(prompt_entry, fail_fast=fail_fast)
        results.append(result)
        if result["success"]:
            print(
                f"  ✓ {result['prompt_id']}: {result['generated_text'][:200]!r}",
                flush=True,
            )
        else:
            print(f"  ✗ {result['prompt_id']}: {result['error']}", flush=True)
            if fail_fast:
                summary_file = OUTPUT_DIR / "sanity_summary.json"
                summary_file.write_text(json.dumps(results, indent=2))
                print(f"\nResults saved to {summary_file}")
                raise SystemExit(1)

    print("\n=== Summary ===")
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"{status} {r['prompt_id']}")

    summary_file = OUTPUT_DIR / "sanity_summary.json"
    summary_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {summary_file}")
