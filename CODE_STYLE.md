# kTransformers Code Style

## Naming Conventions

### Python Files & Modules

| Element | Convention | Example |
|---------|-----------|---------|
| **Files** | `snake_case.py` | `local_chat.py`, `custom_loader.py` |
| **Packages** | `snake_case/` | `ktransformers/`, `optimize_rules/` |
| **Modules** | `snake_case` | `ktransformers.util` |

### Python Classes

| Pattern | Convention | Example |
|---------|-----------|---------|
| **Standard classes** | `PascalCase` | `DeepseekV3Config`, `GGUFLoader` |
| **Wrapper classes** | `KPascalCase` | `KDeepseekV2Attention`, `KExpertsCPU`, `KLinearMarlin` |
| **Cache classes** | `KPascalCase` | `KDeepSeekV3Cache`, `KGQACache` |
| **Config classes** | `PascalCase` | `DeepseekV2Config`, `Qwen2MoeConfig` |

### Python Functions & Variables

| Element | Convention | Example |
|---------|-----------|---------|
| **Functions** | `snake_case()` | `optimize_and_load_gguf()`, `load_weights()` |
| **Variables** | `snake_case` | `gguf_loader`, `tensor_device_map` |
| **Constants** | `SCREAMING_SNAKE_CASE` | `MAX_PAGES`, `DEFAULT_DEVICE` |
| **ML Abbreviations** | `bsz`, `q_len`, `kv` | `bsz, q_len, _ = hidden_states.size()` |

### C++ Naming

| Element | Convention | Example |
|---------|-----------|---------|
| **Classes/Templates** | `PascalCase` | `AMX_MOE_TP`, `MatKernelSelection` |
| **Structs** | `PascalCase` | `GeneralMOEConfig`, `MatKernelSelection` |
| **Member variables** | `snake_case_` (trailing `_`) | `config_`, `down_ba_`, `tp_part_idx` |
| **Functions** | `snake_case()` | `select_kernel_for_int4()`, `forward_prefill()` |
| **Typedefs** | `snake_case_t` | `int4_2_t`, `bfloat16_t` |
| **Enums** | `PascalCase` | `MatKernelVariant` |
| **Enum values** | `PascalCase` | `KernelCblasNoTrans`, `Decode` |

### YAML Configuration

| Element | Convention | Example |
|---------|-----------|---------|
| **Rule files** | `PascalCase.yaml` | `DeepSeek-V3-Chat.yaml` |
| **Keys** | `snake_case` | `generate_device`, `prefill_device` |
| **Values** | varies | `"cuda"`, `8`, `true` |

## File Organization

### Python File Structure

```python
"""
Module docstring with description.
Author: Name
Version: 0.1.0
Copyright (c) YYYY by KVCache.AI, All Rights Reserved.
"""
# Standard library imports (alphabetical)
import logging
from typing import Any, Dict, List, Optional

# Third-party imports
import torch
from transformers import AutoConfig

# Local imports
from ktransformers.util.utils import load_weights


# Module-level logger
logger = logging.getLogger(__name__)


class MyClass:
    """Class docstring."""
    
    def __init__(self, param: str):
        self.param = param
    
    def method(self) -> None:
        """Method docstring."""
        pass


def my_function(arg: int) -> str:
    """Function docstring."""
    return str(arg)
```

### Import Order (PEP 8)

1. `__future__` imports
2. Standard library
3. Third-party
4. Local application/library imports

Separate groups with blank lines.

## Code Patterns

### BaseInjectedModule Pattern

Used for wrapping transformers modules with KTransformers functionality:

```python
from ktransformers.operators.base_operator import BaseInjectedModule

class KMyModule(BaseInjectedModule, OriginalModuleClass):
    """Wrapper that adds KT functionality while preserving original interface."""
    
    def __init__(self, key, gguf_loader, config, orig_module, 
                 prefill_device="cuda", generate_device="cuda", **kwargs):
        super().__init__(key, gguf_loader, config, orig_module,
                        prefill_device, generate_device, **kwargs)
        # Custom initialization
    
    def forward(self, *args, **kwargs):
        # Custom forward or delegate to orig_module
        return self.orig_module.forward(*args, **kwargs)
```

### YAML Rule Pattern

Model-specific optimization rules in `optimize_rules/*.yaml`:

```yaml
# Match by class OR name, replace with optimized implementation
- match:
    class: original.model.ClassName
    # OR
    name: ".*layer_0.*mlp.*"  # regex pattern
  replace:
    class: ktransformers.operators.MyOperator
    kwargs:
      generate_device: "cuda"
      prefill_device: "cuda"
  recursive: true  # Apply to child modules
```

### Factory Pattern (Experts)

```python
class KTMoEWrapper:
    @staticmethod
    def create(method: str, config, tp_part_idx):
        if method == "AMXINT4":
            return AMXMoEWrapper(config, tp_part_idx)
        elif method == "MOE_INT4":
            return GeneralMoEWrapper(config, tp_part_idx)
        else:
            raise ValueError(f"Unknown method: {method}")
```

### Device Transfer Pattern

```python
if self.transfer_map is not None and layer_idx in self.transfer_map:
    hidden_states = hidden_states.to(
        self.transfer_map[layer_idx], 
        non_blocking=True
    )
```

## Error Handling

### Exception Types

| Type | Usage | Example |
|------|-------|---------|
| `ValueError` | Invalid argument values | `raise ValueError(f"Invalid r={r}")` |
| `RuntimeError` | Runtime failures | `raise RuntimeError("Device unavailable")` |
| `NotImplementedError` | Unimplemented features | `raise NotImplementedError(f"Type {dt} not implemented")` |
| `FileNotFoundError` | Missing files | `raise FileNotFoundError(f"Path not found: {path}")` |
| `HTTPException` | Server errors (FastAPI) | `raise HTTPException(500, detail=str(e))` |

### Exception Patterns

```python
# Raising with descriptive message
raise ValueError(f"`r` should be positive, got {r}")

# Catching with logging
try:
    session.commit()
except Exception as e:
    logger.exception("db commit error with data %s", str(data))
    raise

# Exception chaining
raise ValueError("New error") from original_error
```

## Logging

### Logging Setup (3 Systems)

**1. SFT/LLaMA-Factory Logging** (`sft/metrics_utils/logging.py`):
```python
from ktransformers.sft.metrics_utils.logging import get_logger
logger = get_logger(__name__)
logger.info_rank0("Only log on rank 0")  # Distributed training
```

**2. Server Logging** (`server/config/log.py`):
```python
from ktransformers.server.config.log import logger
logger.info("Server started")
```

**3. Simple Logging** (Examples/Tests):
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### Log Levels

| Level | When to Use | Example |
|-------|-------------|---------|
| `DEBUG` | Input shapes, performance stats | `logger.debug(f"input_ids: {input_ids.shape}")` |
| `INFO` | Normal operations, metrics | `logger.info(f'Prefill rate: {rate}')` |
| `WARNING` | Configuration issues, fallbacks | `logger.warning("rope_scaling factor must be >= 1")` |
| `ERROR` | Failures allowing continuation | `logger.error("Unsupported type %s", type)` |
| `exception` | Errors with full traceback | `logger.exception("Commit failed: %s", e)` |

## Testing

### Test File Naming

| Pattern | Meaning |
|---------|---------|
| `test_*.py` | pytest test files |
| `*_test.cpp` | C++ test files |
| `pytest.ini` | pytest configuration |

### pytest Markers

```python
import pytest

@pytest.mark.cpu       # CPU backend tests
@pytest.mark.cuda      # CUDA tests
@pytest.mark.amd       # AMD tests  
@pytest.mark.slow      # Tests >60s
@pytest.mark.requires_model  # Requires model files

def test_moe_forward():
    # Test code
    pass
```

### Test Structure

```python
import pytest
from ktransformers.operators.experts import KExpertsCPU

def test_experts_creation():
    """Test expert creation with mock config."""
    config = MockConfig(expert_num=8)
    experts = KExpertsCPU(config, tp_part_idx=0)
    assert experts.expert_num == 8

@pytest.mark.slow
def test_full_inference():
    """End-to-end inference test (slow)."""
    # Skip on CI if model not available
    pytest.importorskip("transformers")
```

## Do's and Don'ts

### Do

- Use `snake_case` for Python functions/variables
- Use `PascalCase` for Python classes (except K-prefix wrappers)
- Use `logging.getLogger(__name__)` for module loggers
- Add docstrings to classes and public methods
- Use type hints for function parameters and returns
- Handle device placement with `non_blocking=True` for async transfers
- Use f-strings for simple formatting, `%` format for lazy logging

### Don't

- Don't mix `K` prefix inconsistently (it's for KTransformers wrappers)
- Don't use `print()` for debugging (use logger.debug)
- Don't create bare `except:` clauses (specify exception types)
- Don't use `cd` in commands (use `workdir` parameter)
- Don't use `cat`/`head`/`tail` for file operations (use Read tool)

## Tool Configuration

### Black Formatting

```toml
# pyproject.toml
[tool.black]
line-length = 120
target-version = ["py311"]
```

### Ruff Linting

```toml
# pyproject.toml
[tool.ruff]
line-length = 120
target-version = "py311"
```
