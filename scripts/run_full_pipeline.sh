#!/bin/bash
# End-to-end MoE Routing Analysis Pipeline
# Runs: Setup → Collection → Analysis → Simulation → (optional) Real benchmark

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
cd "${ROOT_DIR}"

echo "==============================================="
echo "  MoE Routing Analysis - Full Pipeline"
echo "==============================================="
echo ""

MODEL_PATH="${ROOT_DIR}/models/Qwen3.5-35B-A3B"
GGUF_PATH="${ROOT_DIR}/models/Qwen3.5-35B-A3B-GGUF-Q4_K_M"
GGUF_FILE="${GGUF_PATH}/Qwen3.5-35B-A3B-Q4_K_M.gguf"

# Phase 0: Setup
echo ""
echo "📦 Phase 0: Setup - Ensuring Qwen3.5 models are available..."

if ! command -v hf >/dev/null 2>&1; then
    echo "Error: Hugging Face CLI (hf) not found in PATH."
    echo "Install it first, then re-run this pipeline."
    exit 1
fi

mkdir -p "${ROOT_DIR}/models"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Downloading base model to ${MODEL_PATH} ..."
    hf download "Qwen/Qwen3.5-35B-A3B" --local-dir "${MODEL_PATH}"
else
    echo "✓ Base model directory already exists: ${MODEL_PATH}"
fi

if [ ! -f "${GGUF_FILE}" ]; then
    echo "Downloading GGUF Q4_K_M weights to ${GGUF_PATH} ..."
    hf download \
        "unsloth/Qwen3.5-35B-A3B-GGUF" \
        "Qwen3.5-35B-A3B-Q4_K_M.gguf" \
        --local-dir "${GGUF_PATH}"
else
    echo "✓ GGUF file already exists: ${GGUF_FILE}"
fi

# Phase 1: Data Collection
echo ""
echo "📊 Phase 1: Collecting MoE routing traces..."
if [ -d "data/traces" ] && [ "$(ls -A data/traces/*_session.parquet 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Traces already exist. Skip collection? (y/n)"
    read -r skip
    if [ "$skip" != "y" ]; then
        python3 scripts/run_collection.py
    fi
else
    python3 scripts/run_collection.py
fi

# Phase 2: Analysis
echo ""
echo "🔍 Phase 2: Analyzing locality metrics..."
mkdir -p data/analysis
python -m kt_kernel.moe_routing.analyze \
    --trace-file data/traces/live_capture.parquet \
    --output-dir data/analysis

# Phase 3: Simulation
echo ""
echo "🎯 Phase 3: Simulating cache policies..."
mkdir -p data/simulation
python -m kt_kernel.moe_routing.simulate \
    --trace-file data/traces/live_capture.parquet \
    --output-dir data/simulation

# Phase 4: Real-Inference Benchmark (optional)
echo ""
if [ "${RUN_REAL_BENCHMARK:-0}" = "1" ]; then
    echo "🚀 Phase 4: Running real-inference routing benchmark..."
    python3 scripts/run_real_routing_benchmark.py
else
    echo "⏭️  Phase 4 skipped (set RUN_REAL_BENCHMARK=1 to enable real benchmark)"
fi

echo ""
echo "==============================================="
echo "  Pipeline Complete!"
echo "==============================================="
echo ""
echo "Results:"
echo "  Traces:     data/traces/live_capture.parquet"
echo "  Analysis:   data/analysis/metrics.json"
echo "  Plots:      data/analysis/plots/"
echo "  Simulation: data/simulation/results.json"
echo "  Tradeoffs:  data/simulation/tradeoff_curves.png"
if [ "${RUN_REAL_BENCHMARK:-0}" = "1" ]; then
    echo "  Real bench: data/real_benchmark/results.json"
fi
echo ""
echo "Key metrics to check:"
echo "  - Temporal reuse curve (data/analysis/plots/temporal_reuse_curve.png)"
echo "  - Sliding window hit rate (data/analysis/plots/sliding_window_hit_rate.png)"
echo "  - Tradeoff curves (data/simulation/tradeoff_curves.png)"
