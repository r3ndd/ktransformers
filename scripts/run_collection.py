#!/usr/bin/env python3
"""Full MoE routing collection for Qwen3.5-35B-A3B.

Runs all prompts from data/prompt_suite.json, starts SGLang per prompt with
kt-kernel, captures outputs/traces/logs, and aggregates per-prompt traces into
data/traces/live_capture.parquet.
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
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    os.environ.get(
        "COLLECTION_MODEL_PATH",
        str(ROOT / "models/Qwen3.5-35B-A3B"),
    )
).resolve()

BF16_WEIGHT_PATH = Path(
    os.environ.get(
        "COLLECTION_BF16_WEIGHT_PATH",
        str(MODEL_PATH),
    )
).resolve()

GGUF_WEIGHT_PATH = Path(
    os.environ.get(
        "COLLECTION_GGUF_WEIGHT_PATH",
        str(ROOT / "models/Qwen3.5-35B-A3B-GGUF-Q4_K_M"),
    )
).resolve()

KT_METHOD_ENV = os.environ.get("KT_METHOD", "LLAMAFILE").strip()
# KT_METHOD_ENV = os.environ.get("KT_METHOD", "BF16").strip()

PROMPT_SUITE = Path(
    os.environ.get(
        "COLLECTION_PROMPT_SUITE",
        str(ROOT / "data/prompt_suite.json"),
    )
).resolve()

OUTPUT_DIR = Path(
    os.environ.get(
        "COLLECTION_OUTPUT_DIR",
        str(ROOT / "data/traces"),
    )
).resolve()

PYTHON_BIN = os.environ.get("PYTHON_BIN", "python3")

DEFAULT_PORT = int(os.environ.get("COLLECTION_PORT", "10093"))
KT_CPUINFER_THREADS = int(os.environ.get("COLLECTION_KT_CPUINFER", "1"))
MAX_TOKENS = int(os.environ.get("COLLECTION_MAX_TOKENS", "1024"))
MIN_TOKENS = int(os.environ.get("COLLECTION_MIN_TOKENS", "8"))
SEED = int(os.environ.get("COLLECTION_SEED", "42"))
TEMPERATURE = float(os.environ.get("COLLECTION_TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("COLLECTION_TOP_P", "0.8"))
TOP_K = int(os.environ.get("COLLECTION_TOP_K", "20"))
MIN_P = float(os.environ.get("COLLECTION_MIN_P", "0.0"))
PRESENCE_PENALTY = float(os.environ.get("COLLECTION_PRESENCE_PENALTY", "1.5"))
REPETITION_PENALTY = float(os.environ.get("COLLECTION_REPETITION_PENALTY", "1.0"))
SERVED_MODEL_NAME = os.environ.get("COLLECTION_MODEL_NAME", "Qwen3.5-35B-A3B")
CHAT_TEMPLATE_PATH = Path(
    os.environ.get(
        "COLLECTION_CHAT_TEMPLATE",
        str(MODEL_PATH / "chat_template.jinja"),
    )
).resolve()
VENDORED_SGLANG_PYTHON = (ROOT / "third_party/sglang/python").resolve()

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
    kt_method = KT_METHOD_ENV
    weight_path = ""

    if kt_method == "BF16" and BF16_WEIGHT_PATH.is_dir():
        weight_path = BF16_WEIGHT_PATH
    elif kt_method == "LLAMAFILE" and GGUF_WEIGHT_PATH.is_dir():
        weight_path = GGUF_WEIGHT_PATH
    else:
        raise FileNotFoundError(
            "No CPU weight path found. Please ensure either "
            "COLLECTION_BF16_WEIGHT_PATH or COLLECTION_GGUF_WEIGHT_PATH "
            "is set to a valid directory, or set KT_METHOD to a supported "
            "value (BF16 or LLAMAFILE)." 
        )

    if not weight_path.exists():
        raise FileNotFoundError(f"CPU weights path not found at: {weight_path}")

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

    if not CHAT_TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"Chat template not found at: {CHAT_TEMPLATE_PATH}")

    if not PROMPT_SUITE.is_file():
        raise FileNotFoundError(f"Prompt suite not found at: {PROMPT_SUITE}")

    if not VENDORED_SGLANG_PYTHON.is_dir():
        raise FileNotFoundError(
            f"Vendored SGLang python path not found at: {VENDORED_SGLANG_PYTHON}"
        )

    if not weight_path.is_dir():
        raise FileNotFoundError(f"CPU weights path must be a directory: {weight_path}")


def _load_prompts(
    prompt_id: str | None = None,
    max_prompts: int | None = None,
) -> list[dict]:
    suite = json.loads(_read_text(PROMPT_SUITE))
    prompts = suite.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"No prompts found in suite: {PROMPT_SUITE}")

    if prompt_id:
        matched = [p for p in prompts if p.get("id") == prompt_id]
        if not matched:
            raise ValueError(
                f"Prompt id {prompt_id!r} not found in suite: {PROMPT_SUITE}"
            )
        prompts = matched

    if max_prompts is not None:
        if max_prompts <= 0:
            raise ValueError(f"--max-prompts must be >= 1, got: {max_prompts}")
        prompts = prompts[:max_prompts]

    return prompts


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


def _run_trace_hook_preflight(env: dict[str, str], prompt_id: str, context_id: str) -> None:
    preflight_env = env.copy()
    preflight_env["KT_MOE_ROUTING_PROMPT_ID"] = f"{prompt_id}_preflight"
    preflight_env["KT_MOE_ROUTING_CONTEXT_ID"] = f"{context_id}_preflight"
    preflight_env["KT_MOE_ROUTING_TRACE_FILE"] = str(OUTPUT_DIR / f"{prompt_id}_preflight.parquet")

    probe = (
        "from kt_kernel.experts_base import BaseMoEWrapper\n"
        "BaseMoEWrapper._env_trace_initialized = False\n"
        "BaseMoEWrapper._env_trace_collector = None\n"
        "BaseMoEWrapper.set_trace_hook(None)\n"
        "BaseMoEWrapper._ensure_env_trace_hook()\n"
        "hook = BaseMoEWrapper.get_trace_hook()\n"
        "collector = BaseMoEWrapper._env_trace_collector\n"
        "if hook is None or collector is None:\n"
        "    raise RuntimeError('trace hook/collector not initialized from env')\n"
        "collector.stop()\n"
        "print('TRACE_PREFLIGHT_OK')\n"
    )

    result = subprocess.run(
        [PYTHON_BIN, "-c", probe],
        env=preflight_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or "TRACE_PREFLIGHT_OK" not in result.stdout:
        raise RuntimeError(
            "Trace hook preflight failed. The env-based routing collector did not initialize. "
            f"stdout={result.stdout.strip()!r} stderr={result.stderr.strip()!r}"
        )


def _aggregate_traces(summary_results: list[dict]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:
        print(f"[run_collection] Skipping aggregation (pyarrow unavailable): {e}", flush=True)
        return

    trace_files = [
        Path(r["trace_file"]) for r in summary_results if r.get("success") and Path(r["trace_file"]).exists()
    ]
    if not trace_files:
        print("[run_collection] No successful per-prompt trace files to aggregate.", flush=True)
        return

    tables = [pq.read_table(path) for path in trace_files]
    combined = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
    pq.write_table(combined, AGGREGATED_TRACE_FILE, compression="zstd")
    print(
        f"[run_collection] Aggregated trace saved to {AGGREGATED_TRACE_FILE} "
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
            f"[run_collection] Port {DEFAULT_PORT} busy; using {port} for {prompt_id}.",
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
    existing_pythonpath = env.get("PYTHONPATH", "")
    vendored_path = str(VENDORED_SGLANG_PYTHON)
    if existing_pythonpath:
        path_parts = existing_pythonpath.split(":")
        if vendored_path not in path_parts:
            env["PYTHONPATH"] = f"{vendored_path}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = vendored_path
    env.setdefault(
        "SGLANG_DEBUG_PROMPT_FILE",
        str(OUTPUT_DIR / "debug_prompts.jsonl"),
    )
    env.setdefault("CUDA_HOME", "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime")
    env.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
    env.setdefault("KT_FORCE_CPU_SYNC", "1")

    _run_trace_hook_preflight(env=env, prompt_id=prompt_id, context_id=context_id)
    print("  [preflight] trace hook initialized", flush=True)

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
                "--chat-template",
                str(CHAT_TEMPLATE_PATH),
                "--reasoning-parser",
                "qwen3",
                "--kt-weight-path",
                str(weight_path),
                "--kt-cpuinfer",
                str(KT_CPUINFER_THREADS),
                "--kt-threadpool-count",
                "1",
                "--kt-num-gpu-experts",
                "1",
                "--kt-method",
                kt_method,
                "--kt-gpu-prefill-token-threshold",
                "4096",
                "--kt-max-deferred-experts-per-token",
                "0",
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
                "1",
                "--max-total-tokens",
                "40000",
                "--watchdog-timeout",
                "3000",
                "--random-seed",
                str(SEED),
                "--disable-cuda-graph",
                "--tensor-parallel-size",
                "1",
                "--enable-p2p-check",
                "--disable-shared-experts-fusion",
                "--skip-server-warmup",
            ],
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    result_payload: dict | None = None

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
                    "rid": f"collect_{prompt_id}_{uuid.uuid4().hex[:12]}",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {"role": "user", "content": prompt_text},
                    ],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "separate_reasoning": False,
                    "stream_reasoning": False,
                    "seed": SEED,
                    "max_tokens": MAX_TOKENS,
                    "min_tokens": MIN_TOKENS,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "min_p": MIN_P,
                    "presence_penalty": PRESENCE_PENALTY,
                    "repetition_penalty": REPETITION_PENALTY,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError(f"No choices returned by server. Payload keys: {list(payload.keys())}")

        first_choice = choices[0]
        message = first_choice.get("message") or {}
        generated_text = message.get("content")
        finish_reason = first_choice.get("finish_reason")

        print(
            "  [resp] "
            f"finish_reason={finish_reason!r} ",
            flush=True,
        )

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
                    "raw_response": payload,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(f"  [ok] output: {output_file}", flush=True)
        result_payload = {
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
            raise RuntimeError(f"Collection failed for {prompt_id}: {error_msg}") from e
        return result

    finally:
        _terminate_server(server)
        time.sleep(1)

    if result_payload is not None:
        for _ in range(12):
            if trace_file.exists():
                break
            time.sleep(0.5)
        if not trace_file.exists():
            raise RuntimeError(f"Trace file missing after server shutdown: {trace_file}")
        return result_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen collection run with kt-kernel + sglang")
    parser.add_argument(
        "--prompt-id",
        default=None,
        help="Run only the prompt with this id from the prompt suite",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Run only the first N prompts from the prompt suite",
    )
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
    prompts = _load_prompts(prompt_id=args.prompt_id, max_prompts=args.max_prompts)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if AGGREGATED_TRACE_FILE.exists():
        AGGREGATED_TRACE_FILE.unlink()

    print("=== MoE Routing Collection (kt-kernel + sglang) ===", flush=True)
    print(f"Model: {MODEL_PATH}", flush=True)
    print(f"CPU Weights: {weight_path}", flush=True)
    print(f"KT Method: {kt_method}", flush=True)
    print(f"Python: {PYTHON_BIN}", flush=True)
    print(f"Prompt source: {PROMPT_SUITE}", flush=True)
    print(f"Vendored SGLang PYTHONPATH: {VENDORED_SGLANG_PYTHON}", flush=True)
    print(f"Chat template: {CHAT_TEMPLATE_PATH}", flush=True)
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

    summary_file = OUTPUT_DIR / "capture_summary.json"
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
