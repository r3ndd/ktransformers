"""Tests for prompt_suite_capture module.

These tests focus on helper functions and artifact assembly behavior
without requiring a real model server.
"""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from moe_routing.prompt_suite_capture import (
    PromptSuiteCapture,
    PromptEntry,
    ModelOutput,
    load_prompt_suite,
    aggregate_trace_files,
)
from moe_routing.types import RoutingRecord


def test_load_prompt_suite(tmp_path: Path):
    """Test loading a prompt suite from JSON."""
    suite_data = {
        "description": "Test suite",
        "version": "1.0",
        "prompts": [
            {
                "id": "test_001",
                "category": "coding",
                "description": "Test prompt",
                "prompt": "Write a function to add two numbers.",
            },
            {
                "id": "test_002",
                "category": "reasoning",
                "description": "Another test",
                "prompt": "What is 2+2?",
            },
        ],
    }

    suite_file = tmp_path / "test_suite.json"
    with open(suite_file, "w") as f:
        json.dump(suite_data, f)

    prompts = load_prompt_suite(suite_file)

    assert len(prompts) == 2
    assert prompts[0].id == "test_001"
    assert prompts[0].category == "coding"
    assert prompts[0].prompt == "Write a function to add two numbers."
    assert prompts[1].id == "test_002"
    assert prompts[1].category == "reasoning"


def test_prompt_suite_capture_load(tmp_path: Path):
    """Test PromptSuiteCapture loading."""
    suite_data = {
        "description": "Test suite",
        "version": "1.0",
        "prompts": [
            {
                "id": "p1",
                "category": "coding",
                "description": "Test",
                "prompt": "Hello",
            },
        ],
    }

    suite_file = tmp_path / "suite.json"
    with open(suite_file, "w") as f:
        json.dump(suite_data, f)

    capture = PromptSuiteCapture(
        output_dir=tmp_path / "output",
        prompt_suite_path=suite_file,
    )

    prompts = capture.load_prompt_suite()
    assert len(prompts) == 1
    assert prompts[0].id == "p1"


def test_save_model_output(tmp_path: Path):
    """Test saving model output."""
    suite_data = {
        "description": "Test suite",
        "version": "1.0",
        "prompts": [],
    }
    suite_file = tmp_path / "suite.json"
    with open(suite_file, "w") as f:
        json.dump(suite_data, f)

    capture = PromptSuiteCapture(
        output_dir=tmp_path / "output",
        prompt_suite_path=suite_file,
    )

    output = ModelOutput(
        prompt_id="p1",
        category="coding",
        context_id="ctx_1",
        prompt_text="Hello",
        generated_text="World",
        metadata={"tokens": 5},
    )

    output_file = capture.save_model_output(output)

    assert output_file.exists()
    with open(output_file) as f:
        data = json.load(f)

    assert data["prompt_id"] == "p1"
    assert data["category"] == "coding"
    assert data["context_id"] == "ctx_1"
    assert data["prompt_text"] == "Hello"
    assert data["generated_text"] == "World"
    assert data["metadata"]["tokens"] == 5
    assert "timestamp" in data


def test_aggregate_trace_files(tmp_path: Path):
    """Test aggregating multiple trace files."""
    # Create test trace files
    from moe_routing.parquet_writer import AsyncParquetWriter

    trace1 = tmp_path / "trace1.parquet"
    trace2 = tmp_path / "trace2.parquet"
    output = tmp_path / "aggregated.parquet"

    # Write first trace
    writer1 = AsyncParquetWriter(trace1)
    writer1.start()
    writer1.submit(
        RoutingRecord(
            token_id=1,
            context_id="ctx1",
            layer_id=0,
            token_position=0,
            expert_ids=[1, 2, 3, 4, 5, 6],
            expert_weights=[0.1] * 6,
            timestamp_us=1000,
            token_category="assistant",
        )
    )
    writer1.close()

    # Write second trace
    writer2 = AsyncParquetWriter(trace2)
    writer2.start()
    writer2.submit(
        RoutingRecord(
            token_id=2,
            context_id="ctx2",
            layer_id=1,
            token_position=1,
            expert_ids=[2, 3, 4, 5, 6, 7],
            expert_weights=[0.2] * 6,
            timestamp_us=2000,
            token_category="user",
        )
    )
    writer2.close()

    # Aggregate
    result = aggregate_trace_files([trace1, trace2], output)

    assert result.exists()
    table = pq.read_table(result)
    assert table.num_rows == 2

    # Check that both records are present
    rows = table.to_pylist()
    ctx_ids = {r["context_id"] for r in rows}
    assert ctx_ids == {"ctx1", "ctx2"}


def test_aggregate_trace_files_with_metadata(tmp_path: Path):
    """Test aggregating with prompt metadata."""
    from moe_routing.parquet_writer import AsyncParquetWriter

    trace1 = tmp_path / "trace1.parquet"
    output = tmp_path / "aggregated.parquet"

    writer = AsyncParquetWriter(trace1)
    writer.start()
    writer.submit(
        RoutingRecord(
            token_id=1,
            context_id="ctx_special",
            layer_id=0,
            token_position=0,
            expert_ids=[1, 2, 3, 4, 5, 6],
            expert_weights=[0.1] * 6,
            timestamp_us=1000,
            token_category="assistant",
        )
    )
    writer.close()

    metadata = {
        "ctx_special": {"prompt_id": "p1", "category": "coding"},
    }

    result = aggregate_trace_files([trace1], output, metadata)

    table = pq.read_table(result)
    rows = table.to_pylist()
    assert len(rows) == 1
    assert rows[0]["prompt_id"] == "p1"
    assert rows[0]["category"] == "coding"


def test_capture_prompt_with_generated_text(tmp_path: Path):
    """Test capture_prompt with pre-generated text."""
    suite_data = {
        "description": "Test suite",
        "version": "1.0",
        "prompts": [
            {
                "id": "p1",
                "category": "coding",
                "description": "Test",
                "prompt": "Hello",
            },
        ],
    }
    suite_file = tmp_path / "suite.json"
    with open(suite_file, "w") as f:
        json.dump(suite_data, f)

    capture = PromptSuiteCapture(
        output_dir=tmp_path / "output",
        prompt_suite_path=suite_file,
    )

    entry = PromptEntry(
        id="p1",
        category="coding",
        description="Test",
        prompt="Hello",
    )

    result = capture.capture_prompt(
        prompt_entry=entry,
        context_id="ctx_test",
        generated_text="Generated response",
    )

    assert result.prompt_entry.id == "p1"
    assert result.model_output.generated_text == "Generated response"
    assert result.model_output.context_id == "ctx_test"

    # Check that output was saved
    output_files = list(capture.outputs_dir.glob("*.json"))
    assert len(output_files) == 1


def test_get_outputs_artifact(tmp_path: Path):
    """Test retrieving all outputs."""
    suite_data = {
        "description": "Test suite",
        "version": "1.0",
        "prompts": [],
    }
    suite_file = tmp_path / "suite.json"
    with open(suite_file, "w") as f:
        json.dump(suite_data, f)

    capture = PromptSuiteCapture(
        output_dir=tmp_path / "output",
        prompt_suite_path=suite_file,
    )

    # Create some output files manually
    for i in range(3):
        output = ModelOutput(
            prompt_id=f"p{i}",
            category="test",
            context_id=f"ctx_{i}",
            prompt_text=f"Prompt {i}",
            generated_text=f"Response {i}",
        )
        capture.save_model_output(output)

    outputs = capture.get_outputs_artifact()
    assert len(outputs) == 3

    prompt_ids = {o["prompt_id"] for o in outputs}
    assert prompt_ids == {"p0", "p1", "p2"}


def test_aggregate_traces_empty(tmp_path: Path):
    """Test aggregating with no results."""
    suite_data = {
        "description": "Test suite",
        "version": "1.0",
        "prompts": [],
    }
    suite_file = tmp_path / "suite.json"
    with open(suite_file, "w") as f:
        json.dump(suite_data, f)

    capture = PromptSuiteCapture(
        output_dir=tmp_path / "output",
        prompt_suite_path=suite_file,
    )

    # Aggregate with no captured results
    result = capture.aggregate_traces()

    assert result.exists()
    table = pq.read_table(result)
    assert table.num_rows == 0


def test_prompt_entry_dataclass():
    """Test PromptEntry dataclass."""
    entry = PromptEntry(
        id="test_id",
        category="coding",
        description="A test prompt",
        prompt="Write code",
    )
    assert entry.id == "test_id"
    assert entry.category == "coding"
    assert entry.description == "A test prompt"
    assert entry.prompt == "Write code"


def test_model_output_dataclass():
    """Test ModelOutput dataclass."""
    import time

    before = time.time()
    output = ModelOutput(
        prompt_id="p1",
        category="coding",
        context_id="ctx1",
        prompt_text="Hello",
        generated_text="World",
        metadata={"key": "value"},
    )
    after = time.time()

    assert output.prompt_id == "p1"
    assert output.category == "coding"
    assert output.context_id == "ctx1"
    assert output.prompt_text == "Hello"
    assert output.generated_text == "World"
    assert output.metadata == {"key": "value"}
    assert before <= output.timestamp <= after


def test_aggregate_trace_files_nonexistent_skipped(tmp_path: Path):
    """Test that non-existent trace files are skipped."""
    from moe_routing.parquet_writer import AsyncParquetWriter

    trace1 = tmp_path / "trace1.parquet"
    nonexistent = tmp_path / "nonexistent.parquet"
    output = tmp_path / "aggregated.parquet"

    writer = AsyncParquetWriter(trace1)
    writer.start()
    writer.submit(
        RoutingRecord(
            token_id=1,
            context_id="ctx1",
            layer_id=0,
            token_position=0,
            expert_ids=[1, 2, 3, 4, 5, 6],
            expert_weights=[0.1] * 6,
            timestamp_us=1000,
            token_category="assistant",
        )
    )
    writer.close()

    # Include non-existent file - should not raise
    result = aggregate_trace_files([trace1, nonexistent], output)

    table = pq.read_table(result)
    assert table.num_rows == 1
