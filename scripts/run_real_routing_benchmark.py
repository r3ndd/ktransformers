#!/usr/bin/env python3
"""Run real-inference MoE routing benchmark on the prompt suite.

This script evaluates baseline and routing-scheme sweeps on one long-lived
SGLang server, running requests sequentially (no batching), and reports
real speed + quality-proxy metrics.

Outputs include:
- per-run records with generated text
- per-experiment aggregated metrics
- streaming-derived split timing/throughput metrics (prefill/decode/e2e)
- simulation-like tradeoff metrics (`quality_degradation`, `speedup_ratio`,
  `quality_speed_score`) computed from real inference runs.
"""

from __future__ import annotations

import argparse
import hashlib
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

PROMPT_SUITE = Path(
    os.environ.get(
        "COLLECTION_PROMPT_SUITE",
        str(ROOT / "data/prompt_suite.json"),
    )
).resolve()

OUTPUT_DIR = Path(
    os.environ.get(
        "REAL_BENCHMARK_OUTPUT_DIR",
        str(ROOT / "data/real_benchmark"),
    )
).resolve()

TRACE_DIR = OUTPUT_DIR / "traces"
RUNS_JSONL = OUTPUT_DIR / "runs.jsonl"
RESULTS_JSON = OUTPUT_DIR / "results.json"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
GENERATED_TEXT_JSONL = OUTPUT_DIR / "generated_texts.jsonl"

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
DETERMINISTIC_DEFAULT = os.environ.get("REAL_BENCHMARK_DETERMINISTIC", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DETERMINISTIC_TOKENS = int(os.environ.get("REAL_BENCHMARK_DETERMINISTIC_TOKENS", "256"))
KT_HOT_EXPERT_RATIO = float(os.environ.get("REAL_BENCHMARK_KT_HOT_EXPERT_RATIO", "0.10"))
KT_GPU_SHARE_OF_HOT = float(os.environ.get("REAL_BENCHMARK_KT_GPU_SHARE_OF_HOT", "0.35"))
CHAT_TEMPLATE_PATH = Path(
    os.environ.get(
        "COLLECTION_CHAT_TEMPLATE",
        str(MODEL_PATH / "chat_template.jinja"),
    )
).resolve()
VENDORED_SGLANG_PYTHON = (ROOT / "third_party/sglang/python").resolve()
VENDORED_SGLANG_LAUNCH = (VENDORED_SGLANG_PYTHON / "sglang/launch_server.py").resolve()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_log_tail(log_file: Path, max_lines: int = 120) -> str:
    if not log_file.exists():
        return "<log file not created yet>"
    lines = _read_text(log_file).splitlines()
    return "\n".join(lines[-max_lines:])


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
    if not CHAT_TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"Chat template not found at: {CHAT_TEMPLATE_PATH}")
    if not PROMPT_SUITE.is_file():
        raise FileNotFoundError(f"Prompt suite not found at: {PROMPT_SUITE}")
    if not VENDORED_SGLANG_PYTHON.is_dir():
        raise FileNotFoundError(f"Vendored SGLang python path not found at: {VENDORED_SGLANG_PYTHON}")
    if not VENDORED_SGLANG_LAUNCH.is_file():
        raise FileNotFoundError(f"Vendored SGLang launch_server.py not found at: {VENDORED_SGLANG_LAUNCH}")
    if not weight_path.is_dir():
        raise FileNotFoundError(f"CPU weights path must be a directory: {weight_path}")


def _load_prompts(prompt_id: str | None = None, max_prompts: int | None = None) -> list[dict]:
    suite = json.loads(_read_text(PROMPT_SUITE))
    prompts = suite.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"No prompts found in suite: {PROMPT_SUITE}")

    if prompt_id:
        prompts = [p for p in prompts if p.get("id") == prompt_id]
        if not prompts:
            raise ValueError(f"Prompt id {prompt_id!r} not found in suite: {PROMPT_SUITE}")

    if max_prompts is not None:
        if max_prompts <= 0:
            raise ValueError(f"--max-prompts must be >= 1, got: {max_prompts}")
        prompts = prompts[:max_prompts]
    return prompts


def _read_model_num_experts() -> int:
    cfg_path = MODEL_PATH / "config.json"
    try:
        cfg = json.loads(_read_text(cfg_path))
    except Exception as e:
        raise RuntimeError(f"Failed to parse model config at {cfg_path}: {e}") from e

    text_cfg = cfg.get("text_config") if isinstance(cfg, dict) else None
    if isinstance(text_cfg, dict) and text_cfg.get("num_experts") is not None:
        return int(text_cfg["num_experts"])
    if isinstance(cfg, dict) and cfg.get("num_experts") is not None:
        return int(cfg["num_experts"])
    raise ValueError(f"Could not find num_experts in model config: {cfg_path}")


def _compute_tier_caps(num_experts: int) -> tuple[int, int, int]:
    hot_total = int(round(float(num_experts) * KT_HOT_EXPERT_RATIO))
    hot_total = max(1, min(num_experts, hot_total))

    gpu_cap = int(round(float(hot_total) * KT_GPU_SHARE_OF_HOT))
    gpu_cap = max(1, min(hot_total, gpu_cap))

    cpu_cap = hot_total - gpu_cap
    cpu_cap = max(0, min(num_experts - gpu_cap, cpu_cap))
    return hot_total, gpu_cap, cpu_cap


def _routing_name(prefix: str, cfg: dict) -> str:
    suffix = hashlib.md5(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{suffix}"


def _build_default_experiments() -> list[dict]:
    decode_cfgs: list[dict] = [
        {"scheme": "sliding_window_score_averaging", "params": {"window_size": 1}},
        {"scheme": "sliding_window_score_averaging", "params": {"window_size": 4}},
        {"scheme": "sliding_window_score_averaging", "params": {"window_size": 16}},
        {"scheme": "sliding_window_score_averaging", "params": {"window_size": 64}},
        {"scheme": "ema_score_averaging", "params": {"ema_beta": 0.9}},
        {"scheme": "ema_score_averaging", "params": {"ema_beta": 0.7}},
        {"scheme": "ema_score_averaging", "params": {"ema_beta": 0.5}},
        {"scheme": "ema_score_averaging", "params": {"ema_beta": 0.3}},
        {"scheme": "ema_score_averaging", "params": {"ema_beta": 0.1}},
        {"scheme": "ema_score_averaging", "params": {"ema_beta": 0.05}},
        {"scheme": "two_timescale_ema", "params": {"mix_lambda": 0.1}},
        {"scheme": "two_timescale_ema", "params": {"mix_lambda": 0.2}},
        {"scheme": "two_timescale_ema", "params": {"mix_lambda": 0.3}},
        {"scheme": "two_timescale_ema", "params": {"mix_lambda": 0.4}},
        {"scheme": "two_timescale_softmax", "params": {"mix_lambda": 0.2, "rho": 0.25}},
        {"scheme": "two_timescale_softmax", "params": {"mix_lambda": 0.2, "rho": 1.0}},
        {"scheme": "two_timescale_softmax", "params": {"mix_lambda": 0.2, "rho": 4.0}},
        {"scheme": "two_timescale_softmax", "params": {"mix_lambda": 0.2, "rho": 16.0}},
        {"scheme": "two_timescale_softmax", "params": {"mix_lambda": 0.2, "rho": 64.0}},
        {"scheme": "two_timescale_softmax", "params": {"mix_lambda": 0.2, "rho": 256.0}},
        {"scheme": "two_timescale_softmax", "params": {"mix_lambda": 0.2, "rho": 1024.0}},
    ]
    prefill_cfgs: list[dict] = [
        {"scheme": "prefill_block_mean", "params": {"window_size": 1}},
        {"scheme": "prefill_block_mean", "params": {"window_size": 32}},
        {"scheme": "prefill_block_mean", "params": {"window_size": 64}},
        {"scheme": "prefill_block_mean", "params": {"window_size": 128}},
        {"scheme": "prefill_full_mean", "params": {}},
    ]

    experiments = [{"name": "baseline", "moe_routing": None}]

    for d in decode_cfgs:
        cfg = {
            "prefill": {"scheme": "prefill_block_mean", "params": {"window_size": 1}},
            "decode": d,
            "scope": "request",
        }
        experiments.append({"name": f"decode_{_routing_name(d['scheme'], d)}", "moe_routing": cfg})

    for p in prefill_cfgs:
        cfg = {
            "prefill": p,
            "decode": {"scheme": "sliding_window_score_averaging", "params": {"window_size": 1}},
            "scope": "request",
        }
        experiments.append({"name": f"prefill_{_routing_name(p['scheme'], p)}", "moe_routing": cfg})

    return experiments


def _build_experiments(max_experiments: int | None = None) -> list[dict]:
    custom_json = os.environ.get("REAL_BENCHMARK_EXPERIMENTS_JSON", "").strip()
    if custom_json:
        experiments = json.loads(custom_json)
        if not isinstance(experiments, list):
            raise ValueError("REAL_BENCHMARK_EXPERIMENTS_JSON must be a JSON list")
    else:
        experiments = _build_default_experiments()

    if max_experiments is not None:
        experiments = experiments[:max_experiments]
    return experiments


def _ordered_experiments(experiments: list[dict]) -> list[dict]:
    baseline = [e for e in experiments if str(e.get("name")) == "baseline"]
    others = [e for e in experiments if str(e.get("name")) != "baseline"]
    return baseline + others


def _wait_healthy(port: int, server_proc: subprocess.Popen, log_file: Path, timeout_s: int = 180) -> None:
    for _ in range(timeout_s):
        if server_proc.poll() is not None:
            raise RuntimeError(
                f"Server exited early with code {server_proc.returncode}. Log tail:\n{_read_log_tail(log_file)}"
            )
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return
        except urllib.error.HTTPError:
            time.sleep(1)
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Server failed health check within {timeout_s}s. Log tail:\n{_read_log_tail(log_file)}")


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


def _build_server_env() -> dict[str, str]:
    env = os.environ.copy()
    env["KT_MOE_ROUTING_RECORD"] = "false"
    existing_pythonpath = env.get("PYTHONPATH", "")
    vendored_path = str(VENDORED_SGLANG_PYTHON)
    if existing_pythonpath:
        path_parts = existing_pythonpath.split(":")
        if vendored_path not in path_parts:
            env["PYTHONPATH"] = f"{vendored_path}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = vendored_path
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
    env.setdefault("KT_FORCE_CPU_SYNC", "1")
    return env


def _start_server(
    weight_path: Path,
    kt_method: str,
    port: int,
    log_file: Path,
    *,
    gpu_experts_per_layer: int,
    gpu_cache_capacity: int,
    cpu_cache_capacity: int,
) -> subprocess.Popen:
    env = _build_server_env()
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [
                PYTHON_BIN,
                str(VENDORED_SGLANG_LAUNCH),
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
                str(gpu_experts_per_layer),
                "--kt-method",
                kt_method,
                "--kt-gpu-prefill-token-threshold",
                "4096",
                "--kt-max-deferred-experts-per-token",
                "0",
                "--kt-expert-cache-mode",
                os.environ.get("COLLECTION_KT_EXPERT_CACHE_MODE", "layerwise"),
                "--kt-expert-gpu-cache-capacity",
                os.environ.get("COLLECTION_KT_EXPERT_GPU_CACHE_CAPACITY", str(gpu_cache_capacity)),
                "--kt-expert-cpu-cache-capacity",
                os.environ.get("COLLECTION_KT_EXPERT_CPU_CACHE_CAPACITY", str(cpu_cache_capacity)),
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
    return proc


def _tokenize_simple(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(t for t in text.lower().split() if t)


def _jaccard_similarity(a: str | None, b: str | None) -> float:
    sa = _tokenize_simple(a)
    sb = _tokenize_simple(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def _percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    idx = int(round((p / 100.0) * (len(sorted_values) - 1)))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def _run_single_request(
    port: int,
    prompt_entry: dict,
    experiment: dict,
    run_index: int,
    run_total: int,
    collect_trace: bool,
    deterministic: bool,
) -> dict:
    prompt_id = str(prompt_entry["id"])
    category = str(prompt_entry["category"])
    prompt_text = str(prompt_entry["prompt"])
    exp_name = str(experiment["name"])

    context_id = f"ctx_{prompt_id}_{exp_name}_{uuid.uuid4().hex[:8]}"
    output_file = OUTPUT_DIR / f"output_{prompt_id}_{exp_name}.json"
    trace_file = TRACE_DIR / f"{prompt_id}_{exp_name}_session.parquet"

    if output_file.exists():
        output_file.unlink()
    if trace_file.exists():
        trace_file.unlink()

    print(f"[{run_index}/{run_total}] {prompt_id} [{exp_name}]", flush=True)

    custom_params: dict = {}
    if experiment.get("moe_routing") is not None:
        custom_params["moe_routing"] = experiment["moe_routing"]
    if collect_trace:
        custom_params["moe_trace"] = {
            "output_dir": str(TRACE_DIR),
            "prompt_id": prompt_id,
            "context_id": context_id,
            "token_category": category,
            "trace_file": str(trace_file),
        }

    max_tokens = MAX_TOKENS
    min_tokens = MIN_TOKENS
    temperature = TEMPERATURE
    top_p = TOP_P
    top_k = TOP_K
    min_p = MIN_P
    if deterministic:
        max_tokens = DETERMINISTIC_TOKENS
        min_tokens = DETERMINISTIC_TOKENS
        temperature = 0.0
        top_p = 1.0
        top_k = 1
        min_p = 0.0

    request_payload = {
        "model": SERVED_MODEL_NAME,
        "rid": f"bench_{prompt_id}_{exp_name}_{uuid.uuid4().hex[:10]}",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "separate_reasoning": False,
        "stream_reasoning": False,
        "seed": SEED,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": PRESENCE_PENALTY,
        "repetition_penalty": REPETITION_PENALTY,
        "stream": True,
        "stream_options": {"include_usage": True},
        "custom_params": custom_params or None,
    }

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    t_first_token: float | None = None
    t_end: float | None = None
    generated_parts: list[str] = []
    token_chunk_timestamps: list[float] = []
    finish_reason = None
    usage: dict = {}
    chunk_count = 0
    done_seen = False

    with urllib.request.urlopen(req, timeout=900) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                done_seen = True
                t_end = time.perf_counter()
                break

            chunk_count += 1
            obj = json.loads(data)

            chunk_usage = obj.get("usage")
            if isinstance(chunk_usage, dict):
                usage = chunk_usage

            choices = obj.get("choices") or []
            if choices:
                c0 = choices[0]
                if c0.get("finish_reason") is not None:
                    finish_reason = c0.get("finish_reason")
                delta = c0.get("delta") or {}
                delta_text = delta.get("content")
                if isinstance(delta_text, str) and delta_text:
                    now = time.perf_counter()
                    if t_first_token is None:
                        t_first_token = now
                    token_chunk_timestamps.append(now)
                    generated_parts.append(delta_text)

    if t_end is None:
        t_end = time.perf_counter()
    elapsed_s = float(max(0.0, t_end - t0))
    generated_text = "".join(generated_parts)

    if not done_seen and not generated_text:
        raise RuntimeError("Streaming response ended without any generated content")

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    prefill_latency_s = (float(t_first_token - t0) if t_first_token is not None else None)
    decode_latency_s = (float(t_end - t_first_token) if t_first_token is not None else None)
    e2e_latency_s = elapsed_s

    itl_ms: list[float] = []
    for i in range(1, len(token_chunk_timestamps)):
        itl_ms.append(float((token_chunk_timestamps[i] - token_chunk_timestamps[i - 1]) * 1000.0))
    itl_sorted = sorted(itl_ms)
    itl_mean_ms = (float(sum(itl_ms) / len(itl_ms)) if itl_ms else None)
    itl_p50_ms = _percentile(itl_sorted, 50.0)
    itl_p95_ms = _percentile(itl_sorted, 95.0)

    prefill_toks_per_sec = (
        float(prompt_tokens / prefill_latency_s)
        if prefill_latency_s is not None and prefill_latency_s > 0 and prompt_tokens > 0
        else None
    )
    decode_toks_per_sec = (
        float(completion_tokens / decode_latency_s)
        if decode_latency_s is not None and decode_latency_s > 0 and completion_tokens > 0
        else None
    )
    e2e_toks_per_sec = float(total_tokens / e2e_latency_s) if e2e_latency_s > 0 and total_tokens > 0 else None

    rec = {
        "prompt_id": prompt_id,
        "category": category,
        "experiment": exp_name,
        "moe_routing": experiment.get("moe_routing"),
        "context_id": context_id,
        "seed": SEED,
        "elapsed_seconds": e2e_latency_s,
        "ttft_seconds": prefill_latency_s,
        "ttft_ms": (float(prefill_latency_s * 1000.0) if prefill_latency_s is not None else None),
        "prefill_latency_seconds": prefill_latency_s,
        "decode_latency_seconds": decode_latency_s,
        "e2e_latency_seconds": e2e_latency_s,
        "itl_mean_ms": itl_mean_ms,
        "itl_p50_ms": itl_p50_ms,
        "itl_p95_ms": itl_p95_ms,
        "itl_count": len(itl_ms),
        "decode_tokens_per_second": decode_toks_per_sec,
        "prefill_tokens_per_second": prefill_toks_per_sec,
        "e2e_tokens_per_second": e2e_toks_per_sec,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "finish_reason": finish_reason,
        "generated_text": generated_text,
        "trace_file": str(trace_file) if collect_trace else None,
        "timestamp": time.time(),
    }

    output_file.write_text(
        json.dumps(
            {
                **rec,
                "prompt_text": prompt_text,
                "stream_meta": {
                    "chunks": chunk_count,
                    "done_seen": done_seen,
                    "usage": usage,
                    "finish_reason": finish_reason,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rec["output_file"] = str(output_file)
    rec["success"] = True
    return rec


def _apply_baseline_relative_metrics(rec: dict, baseline: dict | None) -> dict:
    rr = dict(rec)
    if not rr.get("success"):
        return rr

    if baseline is None:
        rr["quality_jaccard"] = None
        rr["quality_degradation"] = None
        rr["speedup_ratio"] = None
        rr["decode_speedup_ratio"] = None
        rr["latency_ratio"] = None
        rr["quality_speed_score"] = None
        rr["exact_text_match"] = None
        return rr

    jac = _jaccard_similarity(rr.get("generated_text"), baseline.get("generated_text"))
    rr["quality_jaccard"] = jac
    rr["quality_degradation"] = 1.0 - jac

    base_tps = float(baseline.get("decode_tokens_per_second") or 0.0)
    cur_tps = float(rr.get("decode_tokens_per_second") or 0.0)
    rr["decode_speedup_ratio"] = (cur_tps / base_tps) if base_tps > 0 and cur_tps > 0 else None

    base_e2e_tps = float(baseline.get("e2e_tokens_per_second") or 0.0)
    cur_e2e_tps = float(rr.get("e2e_tokens_per_second") or 0.0)
    rr["speedup_ratio"] = (cur_e2e_tps / base_e2e_tps) if base_e2e_tps > 0 and cur_e2e_tps > 0 else None

    base_e2e = float(baseline.get("elapsed_seconds") or 0.0)
    cur_e2e = float(rr.get("elapsed_seconds") or 0.0)
    rr["latency_ratio"] = (cur_e2e / base_e2e) if base_e2e > 0 and cur_e2e > 0 else None

    rr["quality_speed_score"] = jac * rr["speedup_ratio"] if rr["speedup_ratio"] is not None else None
    rr["exact_text_match"] = bool((rr.get("generated_text") or "") == (baseline.get("generated_text") or ""))
    return rr


def _aggregate_results(results: list[dict], experiments: list[dict]) -> dict:
    enriched: list[dict] = [dict(r) for r in results]

    exp_runs: dict[str, list[dict]] = {}
    for r in enriched:
        exp_runs.setdefault(str(r.get("experiment")), []).append(r)

    exp_defs = {str(e["name"]): e.get("moe_routing") for e in experiments}
    runs_summary: list[dict] = []

    def _avg(rows: list[dict], key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def _rate(rows: list[dict], key: str) -> float | None:
        vals = [1.0 if r.get(key) else 0.0 for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    for exp_name, rows in exp_runs.items():
        ok_rows = [r for r in rows if r.get("success")]
        run = {
            "experiment": exp_name,
            "moe_routing": exp_defs.get(exp_name),
            "total_runs": len(rows),
            "successful_runs": len(ok_rows),
            "failed_runs": len(rows) - len(ok_rows),
            "avg_prompt_tokens": _avg(ok_rows, "prompt_tokens"),
            "avg_completion_tokens": _avg(ok_rows, "completion_tokens"),
            "avg_elapsed_seconds": _avg(ok_rows, "elapsed_seconds"),
            "avg_ttft_seconds": _avg(ok_rows, "ttft_seconds"),
            "avg_ttft_ms": _avg(ok_rows, "ttft_ms"),
            "avg_prefill_latency_seconds": _avg(ok_rows, "prefill_latency_seconds"),
            "avg_decode_latency_seconds": _avg(ok_rows, "decode_latency_seconds"),
            "avg_e2e_latency_seconds": _avg(ok_rows, "e2e_latency_seconds"),
            "avg_itl_mean_ms": _avg(ok_rows, "itl_mean_ms"),
            "avg_itl_p50_ms": _avg(ok_rows, "itl_p50_ms"),
            "avg_itl_p95_ms": _avg(ok_rows, "itl_p95_ms"),
            "avg_decode_tokens_per_second": _avg(ok_rows, "decode_tokens_per_second"),
            "avg_prefill_tokens_per_second": _avg(ok_rows, "prefill_tokens_per_second"),
            "avg_e2e_tokens_per_second": _avg(ok_rows, "e2e_tokens_per_second"),
            "quality_jaccard": _avg(ok_rows, "quality_jaccard"),
            "quality_degradation": _avg(ok_rows, "quality_degradation"),
            "speedup_ratio": _avg(ok_rows, "speedup_ratio"),
            "decode_speedup_ratio": _avg(ok_rows, "decode_speedup_ratio"),
            "latency_ratio": _avg(ok_rows, "latency_ratio"),
            "quality_speed_score": _avg(ok_rows, "quality_speed_score"),
            "exact_text_match_rate": _rate(ok_rows, "exact_text_match"),
        }
        runs_summary.append(run)

    runs_summary.sort(key=lambda r: (r["experiment"] != "baseline", str(r["experiment"])))
    return {
        "runs": runs_summary,
        "details": enriched,
        "metric_note": {
            "quality_jaccard": "token-set Jaccard similarity against baseline text for same prompt_id and seed",
            "quality_degradation": "1.0 - quality_jaccard",
            "ttft_seconds": "time from request send to first streamed content token",
            "ttft_ms": "ttft_seconds converted to milliseconds",
            "prefill_latency_seconds": "time from request send to first streamed content token",
            "decode_latency_seconds": "time from first streamed content token to end of stream",
            "e2e_latency_seconds": "time from request send to end of stream",
            "itl_mean_ms": "mean inter-token latency (ms) between streamed content chunks",
            "itl_p50_ms": "p50 inter-token latency (ms) between streamed content chunks",
            "itl_p95_ms": "p95 inter-token latency (ms) between streamed content chunks",
            "decode_tokens_per_second": "completion_tokens / decode_latency_seconds",
            "prefill_tokens_per_second": "prompt_tokens / prefill_latency_seconds",
            "e2e_tokens_per_second": "total_tokens / e2e_latency_seconds",
            "speedup_ratio": "e2e_tokens_per_second / baseline_e2e_tokens_per_second for same prompt_id",
            "decode_speedup_ratio": "decode_tokens_per_second / baseline_decode_tokens_per_second for same prompt_id",
            "quality_speed_score": "quality_jaccard * speedup_ratio",
            "latency_ratio": "elapsed_seconds / baseline_elapsed_seconds for same prompt_id",
        },
    }


def _runs_jsonl_row(row: dict) -> dict:
    out = dict(row)
    out.pop("generated_text", None)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-inference MoE routing benchmark")
    parser.add_argument("--prompt-id", default=None, help="Run only one prompt id from suite")
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit prompt count")
    parser.add_argument("--max-experiments", type=int, default=None, help="Limit experiment count")
    parser.add_argument("--collect-traces", action="store_true", help="Also emit per-run parquet traces")
    parser.add_argument("--no-fail-fast", action="store_true", help="Continue on failures")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=DETERMINISTIC_DEFAULT,
        help=(
            "Force deterministic decoding: temperature=0, top_p=1, top_k=1, min_p=0, "
            "and min_tokens=max_tokens=256"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fail_fast = not args.no_fail_fast

    weight_path, kt_method = _resolve_weight_path_and_method()
    _validate_inputs(weight_path)
    prompts = _load_prompts(prompt_id=args.prompt_id, max_prompts=args.max_prompts)
    experiments = _ordered_experiments(_build_experiments(max_experiments=args.max_experiments))
    model_num_experts = _read_model_num_experts()
    hot_total, gpu_cap, cpu_cap = _compute_tier_caps(model_num_experts)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    for p in [RUNS_JSONL, RESULTS_JSON, SUMMARY_JSON, GENERATED_TEXT_JSONL]:
        if p.exists():
            p.unlink()

    print("=== Real-Inference MoE Routing Benchmark ===", flush=True)
    print(f"Model: {MODEL_PATH}", flush=True)
    print(f"CPU Weights: {weight_path}", flush=True)
    print(f"KT Method: {kt_method}", flush=True)
    print(f"Prompt source: {PROMPT_SUITE}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    print(f"Seed: {SEED}", flush=True)
    print(f"Prompts: {len(prompts)}", flush=True)
    print(f"Experiments: {len(experiments)}", flush=True)
    print(f"Collect traces: {args.collect_traces}", flush=True)
    print(f"Deterministic mode: {args.deterministic}", flush=True)
    print(
        f"KT tier caps (per-layer): hot_total={hot_total} ({KT_HOT_EXPERT_RATIO:.1%}), "
        f"gpu={gpu_cap} ({KT_GPU_SHARE_OF_HOT:.1%} of hot), cpu={cpu_cap}, ssd={max(model_num_experts - hot_total, 0)}",
        flush=True,
    )

    port = _pick_available_port(DEFAULT_PORT)
    if port != DEFAULT_PORT:
        print(f"[real_benchmark] Port {DEFAULT_PORT} busy; using {port}.", flush=True)

    server_log_file = OUTPUT_DIR / "benchmark_server.log"
    server = _start_server(
        weight_path=weight_path,
        kt_method=kt_method,
        port=port,
        log_file=server_log_file,
        gpu_experts_per_layer=gpu_cap,
        gpu_cache_capacity=gpu_cap,
        cpu_cache_capacity=cpu_cap,
    )

    results: list[dict] = []
    baseline_by_prompt: dict[str, dict] = {}
    try:
        _wait_healthy(port, server, server_log_file)

        total_runs = len(prompts) * len(experiments)
        run_index = 0
        for prompt in prompts:
            for exp in experiments:
                run_index += 1
                try:
                    raw_rec = _run_single_request(
                        port=port,
                        prompt_entry=prompt,
                        experiment=exp,
                        run_index=run_index,
                        run_total=total_runs,
                        collect_trace=args.collect_traces,
                        deterministic=args.deterministic,
                    )
                    prompt_id = str(raw_rec["prompt_id"])
                    if str(raw_rec.get("experiment")) == "baseline":
                        baseline_by_prompt[prompt_id] = raw_rec
                    baseline = baseline_by_prompt.get(prompt_id)
                    rec = _apply_baseline_relative_metrics(raw_rec, baseline)
                    results.append(rec)
                    with RUNS_JSONL.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(_runs_jsonl_row(rec), ensure_ascii=False) + "\n")
                    with GENERATED_TEXT_JSONL.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "prompt_id": rec["prompt_id"],
                                    "experiment": rec["experiment"],
                                    "seed": rec["seed"],
                                    "generated_text": rec.get("generated_text"),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                except Exception as e:
                    err = {
                        "prompt_id": str(prompt.get("id", "<unknown>")),
                        "category": str(prompt.get("category", "unknown")),
                        "experiment": str(exp.get("name", "<unknown>")),
                        "moe_routing": exp.get("moe_routing"),
                        "success": False,
                        "error": str(e),
                        "server_log": str(server_log_file),
                    }
                    results.append(err)
                    with RUNS_JSONL.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(err, ensure_ascii=False) + "\n")
                    print(f"[err] {err['prompt_id']} [{err['experiment']}]: {err['error']}", flush=True)
                    if fail_fast:
                        raise RuntimeError(
                            f"Benchmark failed for {err['prompt_id']} [{err['experiment']}]: {err['error']}"
                        ) from e
    finally:
        _terminate_server(server)
        time.sleep(1)

    agg = _aggregate_results(results, experiments)

    RESULTS_JSON.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "total": len(results),
        "successful": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "prompts": len(prompts),
        "experiments": len(experiments),
        "seed": SEED,
        "output_files": {
            "runs_jsonl": str(RUNS_JSONL),
            "generated_texts_jsonl": str(GENERATED_TEXT_JSONL),
            "results_json": str(RESULTS_JSON),
            "server_log": str(server_log_file),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Summary ===", flush=True)
    print(f"Successful: {summary['successful']} / {summary['total']}", flush=True)
    print(f"Results: {RESULTS_JSON}", flush=True)
    print(f"Generated texts: {GENERATED_TEXT_JSONL}", flush=True)
    print(f"Run records: {RUNS_JSONL}", flush=True)

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
