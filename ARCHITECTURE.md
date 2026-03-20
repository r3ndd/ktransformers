# kTransformers Architecture

## Overview

kTransformers is a **monorepo** for heterogeneous CPU-GPU LLM inference and fine-tuning. It provides high-performance kernel operations (kt-kernel) and an integrated fine-tuning framework (kt-sft) for ultra-large MoE models.

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Languages** | Python, C++, CUDA |
| **ML Framework** | PyTorch 2.3+, Transformers 4.51.3 |
| **Kernels** | Intel AMX, AVX-512, AVX2, Triton |
| **Quantization** | INT4, INT8, FP8, GPTQ/Marlin |
| **Server** | FastAPI, Uvicorn |
| **Build** | CMake, setuptools, pybind11 |
| **Fine-tuning** | LLaMA-Factory integration, PEFT |

## Directory Structure

```
ktransformers/
├── kt-kernel/                 # High-performance inference kernels
│   ├── python/                # Python package (kt_kernel)
│   │   ├── cli/              # CLI tool (main.py + commands/)
│   │   └── utils/            # Utility functions
│   ├── operators/            # C++ kernel implementations
│   │   ├── amx/             # Intel AMX optimized kernels
│   │   ├── moe_kernel/      # MoE-specific kernels
│   │   └── llamafile/       # Llamafile CPU backend
│   ├── cpu_backend/         # CPU inference backend
│   ├── cuda/                # CUDA kernels
│   ├── scripts/             # Utility scripts (convert weights)
│   └── test/                # Test suite
│
├── kt-sft/                   # Fine-tuning framework
│   ├── ktransformers/        # Main Python package
│   │   ├── operators/        # Custom operators (experts, attention, RoPE)
│   │   ├── models/           # Model implementations
│   │   ├── optimize/         # YAML-based optimization rules
│   │   │   └── optimize_rules/  # Model-specific configs
│   │   ├── sft/              # SFT modules (LoRA, metrics)
│   │   ├── server/           # Inference server
│   │   ├── util/             # Utilities (weight loader, globals)
│   │   └── tests/            # Unit tests
│   └── csrc/                 # C++ extensions
│
├── archive/                   # Legacy original code (read-only)
├── third_party/               # Vendored dependencies
│   ├── sglang/               # SGLang framework
│   ├── llama.cpp/            # llama.cpp backend
│   └── custom_flashinfer/    # Custom FlashInfer kernels
└── doc/                       # Documentation (en/zh)
```

## Core Components

### 1. kt-kernel (Inference Kernels)

**Purpose**: CPU-optimized MoE inference kernels with AMX/AVX/CUDA support.

**Entry Point**: `kt_kernel.cli.main:main()` (CLI command: `kt`)

**Key Modules**:
| Module | Purpose |
|--------|---------|
| `operators/amx/` | Intel AMX BF16/INT4/INT8 MoE kernels |
| `operators/moe_kernel/` | Cross-platform int8/int4 kernels (oneDNN) |
| `operators/kvcache/` | KV cache operations |
| `python/experts.py` | Factory for KTMoEWrapper dispatch |
| `python/experts_base.py` | Base wrapper with CPU inference engine |

**Kernel Backend Selection**:
```python
KTMoEWrapper(method)  # -> AMXMoEWrapper, GeneralMoEWrapper, LlamafileMoEWrapper
```

### 2. kt-sft (Fine-Tuning Framework)

**Purpose**: KTransformers × LLaMA-Factory integration for heterogeneous MoE fine-tuning.

**Entry Point**: `ktransformers.server.main:main()` (CLI command: `ktransformers`)

**Key Modules**:
| Module | Purpose |
|--------|---------|
| `optimize/optimize.py` | YAML rule-based module injection |
| `operators/base_operator.py` | BaseInjectedModule wrapper pattern |
| `operators/experts.py` | KExpertsCPU, KExpertsGPU (MoE routing) |
| `operators/attention.py` | Custom attention with flashinfer |
| `models/custom_*.py` | Custom model implementations (DeepSeek, Qwen MoE) |
| `local_chat.py` | End-to-end inference entry point |

### 3. Optimization System (YAML-based)

**Purpose**: Dynamically replaces model modules with optimized implementations.

**Entry Point**: `optimize_and_load_gguf()` at `optimize/optimize.py:117`

**Data Flow**:
```
1. Load YAML rule file (match/replace patterns)
2. Traverse model hierarchy via gen_optimize_config()
3. Match modules by class type or name regex
4. inject() replaces matched modules with K-prefixed wrappers
5. load_weights() loads from GGUF/safetensors
6. del_meta() cleans up meta tensors
```

**Example Rule** (`DeepSeek-V3-Chat.yaml`):
```yaml
- match:
    class: ktransformers.models.modeling_deepseek_v3.DeepseekV3RotaryEmbedding
  replace:
    class: ktransformers.operators.RoPE.YarnRotaryEmbeddingV3
    kwargs:
      generate_device: "cuda"
      prefill_device: "cuda"
```

### 4. BaseInjectedModule Pattern

**Purpose**: Transparent wrapper preserving original module interface while adding KTransformers functionality.

**Location**: `operators/base_operator.py`

**Key Methods**:
| Method | Purpose |
|--------|---------|
| `__getattr__()` | Three-tier fallback: object attrs → orig_module → orig_module attributes |
| `__setattr__()` | Routes writes to appropriate target |
| `forward()` | Passes through to orig_module |
| `load()` | Loads weights for child modules |

**Pattern**:
```python
class KDeepseekV2Attention(BaseInjectedModule, DeepseekV2Attention):
    def __init__(self, key, gguf_loader, config, orig_module, ...):
        super().__init__(key, gguf_loader, config, orig_module, ...)
        # Custom initialization
```

## Data Flow

### Inference Pipeline

```
User Input (tokens)
    │
    ▼
local_chat()                    # Entry: local_chat.py:87
    │
    ├── Load tokenizer + config
    ├── Create model (meta device)
    ├── optimize_and_load_gguf()  # Module injection + weight loading
    │       │
    │       ├── gen_optimize_config()  # Parse YAML rules
    │       ├── inject()  # Replace modules
    │       └── load_weights()  # Load from GGUF/safetensors
    │
    ▼
prefill_and_generate()          # util/utils.py:344
    │
    ├── chunk_prefill()  # Process input in chunks
    │       │
    │       ├── Embed tokens (CPU → GPU)
    │       ├── FlashInfer MLA attention (GPU)
    │       └── KExpertsCPU (MoE experts on CPU)
    │
    └── Generation loop
            │
            ├── decode_one_tokens()  # CUDA graph or normal
            ├── KExpertsCPU forward  # CPU MoE
            └── Stream output
```

### Weight Loading

```
GGUF/Safetensor file
    │
    ▼
ModelLoaderFactory.create_loader()  # Creates GGUFLoader or SafeTensorLoader
    │
    ▼
load_weights()  # util/utils.py:277
    │
    ├── If BaseInjectedModule: delegate to .load()
    └── Else: load_cur_state_dict() + recurse children
```

## External Integrations

| Integration | Purpose |
|-------------|---------|
| **LLaMA-Factory** | SFT integration via `USE_KT=1` |
| **SGLang** | Server framework (kvcache-ai fork) |
| **FlashInfer** | MLA and GQA attention kernels |
| **llama.cpp** | CPU inference backend |
| **PEFT** | LoRA adapter support |
| **Transformers** | Model loading and tokenization |

## Configuration

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `USE_KT` | Enable KTransformers integration | - |
| `LOCAL_RANK` | Distributed training rank | `0` |
| `KT_OPTIMIZE_RULE` | Path to YAML optimize rule | Auto |
| `LLAMAFACTORY_VERBOSITY` | Log level | `INFO` |

### YAML Optimization Rules

Location: `kt-sft/ktransformers/optimize/optimize_rules/`

Files follow pattern: `{ModelName}.yaml` (e.g., `DeepSeek-V3-Chat.yaml`)

## Build & Deploy

### Build kt-kernel

```bash
cd kt-kernel
pip install -e .
# Builds C++ extensions with AMX/AVX support
```

### Build kt-sft

```bash
cd kt-sft
pip install -e .
# Builds C++ extensions + integrates with LLaMA-Factory
```

### Docker

```bash
# CPU inference
docker build -f kt-sft/Dockerfile -t ktransformers .

# Intel GPU (XPU)
docker build -f kt-sft/Dockerfile.xpu -t ktransformers:xpu .
```

## Testing

### Python Tests (pytest)

```bash
# Run kt-kernel tests
cd kt-kernel && pytest test/

# Markers
@pytest.mark.cpu    # CPU backend
@pytest.mark.cuda   # CUDA backend
@pytest.mark.amd    # AMD backend
@pytest.mark.slow   # >60s tests
```

### C++ Tests

```bash
# Compile and run C++ tests
cd kt-kernel/operators/amx/test
cmake .. && make && ./amx-test
```
