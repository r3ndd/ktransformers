#!/bin/bash
# End-to-end MoE Routing Analysis Pipeline
# Runs: Setup → Collection → Analysis → Simulation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "==============================================="
echo "  MoE Routing Analysis - Full Pipeline"
echo "==============================================="
echo ""

# Phase 0: Setup
if [ ! -f "models/DeepSeek-V2-Lite-Chat-GGUF/DeepSeek-V2-Lite-Chat.Q4_K_M.gguf" ]; then
    echo "📦 Phase 0: Setup - Downloading model..."
    ./scripts/setup_deepseek_v2_lite.sh
else
    echo "✓ Model already downloaded"
fi

# Phase 1: Data Collection
echo ""
echo "📊 Phase 1: Collecting MoE routing traces..."
if [ -d "data/traces" ] && [ "$(ls -A data/traces/*.parquet 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Traces already exist. Skip collection? (y/n)"
    read -r skip
    if [ "$skip" != "y" ]; then
        ./scripts/run_collection.sh
    fi
else
    ./scripts/run_collection.sh
fi

# Phase 2: Analysis
echo ""
echo "🔍 Phase 2: Analyzing locality metrics..."
mkdir -p data/analysis
python -m kt_kernel.moe_routing.analyze \
    --trace-file data/traces/*.parquet \
    --output-dir data/analysis

# Phase 3: Simulation
echo ""
echo "🎯 Phase 3: Simulating cache policies..."
mkdir -p data/simulation
python -m kt_kernel.moe_routing.simulate \
    --trace-file data/traces/*.parquet \
    --output-dir data/simulation

echo ""
echo "==============================================="
echo "  Pipeline Complete!"
echo "==============================================="
echo ""
echo "Results:"
echo "  Traces:     data/traces/*.parquet"
echo "  Analysis:   data/analysis/metrics.json"
echo "  Plots:      data/analysis/plots/"
echo "  Simulation: data/simulation/results.json"
echo "  Tradeoffs:  data/simulation/tradeoff_curves.png"
echo ""
echo "Key metrics to check:"
echo "  - Temporal reuse curve (data/analysis/plots/temporal_reuse_curve.png)"
echo "  - Sliding window hit rate (data/analysis/plots/sliding_window_hit_rate.png)"
echo "  - Tradeoff curves (data/simulation/tradeoff_curves.png)"
