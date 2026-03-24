#!/usr/bin/env python3
"""Sanity check for Qwen3.5-35B-A3B with run_collection.sh assumptions.

Runs prompt suite entries one-by-one, starts SGLang per prompt with kt-kernel,
captures outputs/traces/logs, and fails fast on SIGQUIT-style child failures.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    os.environ.get(
        "SANITY_MODEL_PATH",
        str(ROOT / "models/Qwen3.5-35B-A3B"),
    )
).resolve()

GGUF_PATH = Path(
    os.environ.get(
        "SANITY_GGUF_PATH",
        str(ROOT / "models/Qwen3.5-35B-A3B-GGUF-Q4_K_M"),
    )
).resolve()

PROMPT_SUITE = Path(
    os.environ.get(
        "SANITY_PROMPT_SUITE",
        str(ROOT / "data/prompt_suite.json"),
    )
).resolve()

OUTPUT_DIR = Path(
    os.environ.get(
        "SANITY_OUTPUT_DIR",
        str(ROOT / "data/traces"),
    )
).resolve()

PYTHON_BIN = os.environ.get("PYTHON_BIN", "python3")

DEFAULT_PORT = int(os.environ.get("SANITY_PORT", "10093"))
MAX_TOKENS = int(os.environ.get("SANITY_MAX_TOKENS", "64"))
TEMPERATURE = float(os.environ.get("SANITY_TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("SANITY_TOP_P", "0.9"))
SERVED_MODEL_NAME = os.environ.get("SANITY_MODEL_NAME", "Qwen3.5-35B-A3B")
MAX_PROMPTS = int(os.environ.get("SANITY_MAX_PROMPTS", "1"))

KT_WEIGHT_PATH_ENV = os.environ.get("KT_WEIGHT_PATH", "").strip()
KT_METHOD_ENV = os.environ.get("KT_METHOD", "").strip()

AGGREGATED_TRACE_FILE = OUTPUT_DIR / "live_capture.parquet"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_log_tail(log_file: Path, max_lines: int = 120) -> str:
    if not log_file.exists():
        return "<log file not created yet>"
    lines = _read_text(log_file).splitlines()
    return "\n".join(lines[-max_lines:])


def _log_has_sigquit_failure(log_file: Path) -> tuple[bool, str | None]:
    if not log_file.exists():
        return False, None

    text = _read_text(log_file)
    markers = [
        "Received sigquit from a child process",
        "SIGQUIT received",
        "Scheduler hit an exception",
    ]
    for marker in markers:
        if marker in text:
            return True, marker
    return False, None


def _pick_available_port(preferred_port: int) -> int:
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


def _resolve_weight_path_and_method() -> tuple[Path, str]:
    weight_path = Path(KT_WEIGHT_PATH_ENV).resolve() if KT_WEIGHT_PATH_ENV else None
    kt_method = KT_METHOD_ENV if KT_METHOD_ENV else None

    if weight_path is None:
        if GGUF_PATH.is_dir():
            weight_path = GGUF_PATH
            kt_method = kt_method or "LLAMAFILE"
        else:
            raise FileNotFoundError(
                "No CPU weight path found. Expected GGUF directory at "
                f"{GGUF_PATH}, or provide KT_WEIGHT_PATH and KT_METHOD."
            )

    if not weight_path.exists():
        raise FileNotFoundError(f"CPU weights path not found at: {weight_path}")

    if kt_method is None:
        if "gguf" in str(weight_path).lower():
            kt_method = "LLAMAFILE"
        else:
            kt_method = "AMXINT4"

    return weight_path, kt_method


def _validate_inputs(weight_path: Path) -> None:
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    if not (MODEL_PATH / "model.safetensors.index.json").exists():
        raise FileNotFoundError(
            f"Missing model.safetensors.index.json in model path: {MODEL_PATH}"
        )

    if not any(MODEL_PATH.glob("*.safetensors")):
        raise FileNotFoundError(f"Missing *.safetensors shards in model path: {MODEL_PATH}")

    if not PROMPT_SUITE.is_file():
        raise FileNotFoundError(f"Prompt suite not found at: {PROMPT_SUITE}")

    if not weight_path.is_dir():
        raise FileNotFoundError(f"CPU weights path must be a directory: {weight_path}")


def _load_prompts() -> list[dict]:
    suite = json.loads(_read_text(PROMPT_SUITE))
    prompts = suite.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"No prompts found in suite: {PROMPT_SUITE}")
    if MAX_PROMPTS <= 0:
        raise ValueError(f"SANITY_MAX_PROMPTS must be >= 1, got: {MAX_PROMPTS}")
    return prompts[:MAX_PROMPTS]


def _wait_healthy(port: int, server_proc: subprocess.Popen, log_file: Path, timeout_s: int = 180) -> None:
    for _ in range(timeout_s):
        if server_proc.poll() is not None:
            raise RuntimeError(
                f"Server exited early with code {server_proc.returncode}. "
                f"Log tail:\n{_read_log_tail(log_file)}"
            )

        sigquit_failed, marker = _log_has_sigquit_failure(log_file)
        if sigquit_failed:
            raise RuntimeError(
                f"Detected SIGQUIT child failure ({marker}). "
                f"Log tail:\n{_read_log_tail(log_file)}"
            )

        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return
        except urllib.error.HTTPError:
            time.sleep(1)
        except Exception:
            time.sleep(1)

    raise RuntimeError(
        f"Server failed health check within {timeout_s}s. "
        f"Log tail:\n{_read_log_tail(log_file)}"
    )


def _terminate_server(server: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(server.pid), signal.SIGINT)
    except Exception:
        pass
    try:
        server.wait(timeout=30)
        return
    except Exception:
        pass

    try:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        server.wait(timeout=20)
        return
    except Exception:
        pass

    try:
        os.killpg(os.getpgid(server.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        server.wait(timeout=10)
    except Exception:
        pass


def _aggregate_traces(summary_results: list[dict]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:
        print(f"[sanity_check] Skipping aggregation (pyarrow unavailable): {e}", flush=True)
        return

    trace_files = [
        Path(r["trace_file"]) for r in summary_results if r.get("success") and Path(r["trace_file"]).exists()
    ]
    if not trace_files:
        print("[sanity_check] No successful per-prompt trace files to aggregate.", flush=True)
        return

    tables = [pq.read_table(path) for path in trace_files]
    combined = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
    pq.write_table(combined, AGGREGATED_TRACE_FILE, compression="zstd")
    print(
        f"[sanity_check] Aggregated trace saved to {AGGREGATED_TRACE_FILE} "
        f"({combined.num_rows} rows from {len(trace_files)} files)",
        flush=True,
    )


def run_prompt(
    prompt_entry: dict,
    idx: int,
    total: int,
    weight_path: Path,
    kt_method: str,
    fail_fast: bool,
) -> dict:
    prompt_id = prompt_entry["id"]
    category = prompt_entry["category"]
    prompt_text = prompt_entry["prompt"]
    context_id = f"ctx_{prompt_id}"

    trace_file = OUTPUT_DIR / f"{prompt_id}_session.parquet"
    output_file = OUTPUT_DIR / f"output_{prompt_id}.json"
    log_file = OUTPUT_DIR / f"collection_server_{prompt_id}.log"

    for old in (trace_file, output_file):
        if old.exists():
            old.unlink()

    port = _pick_available_port(DEFAULT_PORT)
    if port != DEFAULT_PORT:
        print(
            f"[sanity_check] Port {DEFAULT_PORT} busy; using {port} for {prompt_id}.",
            flush=True,
        )

    print(f"[{idx}/{total}] Processing prompt: {prompt_id} (category: {category})", flush=True)

    env = os.environ.copy()
    env["KT_MOE_ROUTING_RECORD"] = "true"
    env["KT_MOE_ROUTING_TRACE_DIR"] = str(OUTPUT_DIR)
    env["KT_MOE_ROUTING_TRACE_FILE"] = str(trace_file)
    env["KT_MOE_ROUTING_PROMPT_ID"] = prompt_id
    env["KT_MOE_ROUTING_CONTEXT_ID"] = context_id
    env["KT_MOE_ROUTING_TOKEN_CATEGORY"] = category
    env.setdefault("CUDA_HOME", "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime")
    env.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")

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
                str(weight_path),
                "--kt-cpuinfer",
                "25",
                "--kt-threadpool-count",
                "1",
                "--kt-num-gpu-experts",
                "1",
                "--kt-method",
                kt_method,
                "--kt-gpu-prefill-token-threshold",
                "4096",
                "--kt-enable-dynamic-expert-update",
                "--attention-backend",
                "triton",
                "--sampling-backend",
                "pytorch",
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
                "--disable-cuda-graph",
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
        _wait_healthy(port, server, log_file)

        sigquit_failed, marker = _log_has_sigquit_failure(log_file)
        if sigquit_failed:
            raise RuntimeError(
                f"Detected SIGQUIT child failure before request ({marker}). "
                f"Log tail:\n{_read_log_tail(log_file)}"
            )

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": SERVED_MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
        generated_text = payload["choices"][0]["message"]["content"]

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
            raise RuntimeError(f"Trace file missing: {trace_file}")

        print(f"  [ok] output: {output_file}", flush=True)
        return {
            "prompt_id": prompt_id,
            "output_file": str(output_file),
            "trace_file": str(trace_file),
            "server_log": str(log_file),
            "success": True,
        }

    except Exception as e:
        sigquit_failed, marker = _log_has_sigquit_failure(log_file)
        error_msg = str(e)
        if sigquit_failed:
            error_msg = (
                f"SIGQUIT child-process failure detected ({marker}). "
                f"Error: {e}"
            )

        print(f"  [err] {prompt_id}: {error_msg}", flush=True)
        print(f"  [err] log tail for {prompt_id}:\n{_read_log_tail(log_file)}", flush=True)

        result = {
            "prompt_id": prompt_id,
            "trace_file": str(trace_file),
            "server_log": str(log_file),
            "error": error_msg,
            "success": False,
        }

        if fail_fast:
            raise RuntimeError(f"Sanity check failed for {prompt_id}: {error_msg}") from e
        return result

    finally:
        _terminate_server(server)
        time.sleep(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen sanity check with run_collection assumptions")
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Continue on prompt failures instead of exiting immediately",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fail_fast = not args.no_fail_fast

    weight_path, kt_method = _resolve_weight_path_and_method()
    _validate_inputs(weight_path)
    prompts = _load_prompts()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if AGGREGATED_TRACE_FILE.exists():
        AGGREGATED_TRACE_FILE.unlink()

    print("=== MoE Routing Sanity Check (kt-kernel + sglang) ===", flush=True)
    print(f"Model: {MODEL_PATH}", flush=True)
    print(f"CPU Weights: {weight_path}", flush=True)
    print(f"KT Method: {kt_method}", flush=True)
    print(f"Python: {PYTHON_BIN}", flush=True)
    print(f"Prompt source: {PROMPT_SUITE}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    print(f"Aggregated trace: {AGGREGATED_TRACE_FILE}", flush=True)
    print(f"Total prompts to process: {len(prompts)}", flush=True)

    results: list[dict] = []
    for index, prompt in enumerate(prompts, start=1):
        try:
            result = run_prompt(
                prompt_entry=prompt,
                idx=index,
                total=len(prompts),
                weight_path=weight_path,
                kt_method=kt_method,
                fail_fast=fail_fast,
            )
            results.append(result)
        except Exception as e:
            results.append(
                {
                    "prompt_id": prompt.get("id", f"prompt_{index}"),
                    "success": False,
                    "error": str(e),
                }
            )
            if fail_fast:
                break

    summary = {
        "total": len(prompts),
        "successful": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "results": results,
    }

    summary_file = OUTPUT_DIR / "sanity_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Summary ===", flush=True)
    for r in results:
        status = "[ok]" if r.get("success") else "[x]"
        print(f"{status} {r.get('prompt_id', '<unknown>')}", flush=True)
    print(f"Summary saved to {summary_file}", flush=True)

    _aggregate_traces(results)

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
