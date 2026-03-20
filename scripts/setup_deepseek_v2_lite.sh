#!/bin/bash
# Setup script for DeepSeek-V2-Lite with MoE routing analysis
# Run from repository root

set -e

echo "=== Setting up DeepSeek-V2-Lite for MoE Routing Analysis ==="

# Create directories
mkdir -p data/traces
mkdir -p data/analysis
mkdir -p data/simulation
mkdir -p models/DeepSeek-V2-Lite-Chat-GGUF

# Download GGUF model if not present
if [ ! -f "models/DeepSeek-V2-Lite-Chat-GGUF/DeepSeek-V2-Lite-Chat.Q4_K_M.gguf" ]; then
    echo "Downloading DeepSeek-V2-Lite Q4_K_M GGUF model..."
    wget -O models/DeepSeek-V2-Lite-Chat-GGUF/DeepSeek-V2-Lite-Chat.Q4_K_M.gguf \
        "https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/main/DeepSeek-V2-Lite.Q4_K_M.gguf"
else
    echo "Model already downloaded."
fi

# Download model config from HuggingFace if not present
if [ ! -d "models/deepseek-ai/DeepSeek-V2-Lite-Chat" ]; then
    echo "Downloading model config files..."
    mkdir -p models/deepseek-ai
    cd models/deepseek-ai
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat
    cd ../..
else
    echo "Model config already present."
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q pyarrow pandas matplotlib seaborn

echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Run inference with trace collection:"
echo "   ./scripts/run_collection.sh"
echo ""
echo "2. Analyze collected traces:"
echo "   python -m kt_kernel.moe_routing.analyze --trace-file data/traces/*.parquet --output-dir data/analysis"
echo ""
echo "3. Simulate cache policies:"
echo "   python -m kt_kernel.moe_routing.simulate --trace-file data/traces/*.parquet --output-dir data/simulation"
