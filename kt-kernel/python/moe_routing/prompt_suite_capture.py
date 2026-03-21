"""Prompt suite capture functionality for MoE routing trace collection.

This module provides utilities to capture traces across all prompts in a prompt suite,
persist per-prompt outputs, and aggregate traces for downstream analysis.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from .trace_collector import RoutingTraceCollector
from .types import RoutingRecord


@dataclass
class PromptEntry:
    """A single prompt entry from the prompt suite."""

    id: str
    category: str
    description: str
    prompt: str


@dataclass
class ModelOutput:
    """Model output for a single prompt."""

    prompt_id: str
    category: str
    context_id: str
    prompt_text: str
    generated_text: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaptureResult:
    """Result of capturing a single prompt."""

    prompt_entry: PromptEntry
    model_output: ModelOutput
    trace_file: Path | None


class PromptSuiteCapture:
    """Capture traces for all prompts in a prompt suite.

    This class handles:
    1. Loading the prompt suite from JSON
    2. Iterating through all prompts
    3. Capturing traces for each prompt
    4. Saving per-prompt model outputs
    5. Aggregating traces into a combined parquet file
    """

    def __init__(
        self,
        output_dir: Path,
        prompt_suite_path: Path,
        generate_fn: Callable[[str], str] | None = None,
    ):
        """Initialize the prompt suite capture.

        Args:
            output_dir: Directory to save outputs and traces
            prompt_suite_path: Path to the prompt suite JSON file
            generate_fn: Optional function to generate model output from prompt text.
                        If not provided, the caller must handle generation externally.
        """
        self.output_dir = Path(output_dir)
        self.prompt_suite_path = Path(prompt_suite_path)
        self.generate_fn = generate_fn

        # Create subdirectories
        self.traces_dir = self.output_dir / "traces"
        self.outputs_dir = self.output_dir / "outputs"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Track capture results
        self.results: list[CaptureResult] = []
        self._current_collector: RoutingTraceCollector | None = None
        self._current_trace_file: Path | None = None

    def load_prompt_suite(self) -> list[PromptEntry]:
        """Load the prompt suite from JSON file.

        Returns:
            List of PromptEntry objects

        Raises:
            FileNotFoundError: If prompt suite file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        with open(self.prompt_suite_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        prompts = []
        for p in data.get("prompts", []):
            prompts.append(
                PromptEntry(
                    id=p["id"],
                    category=p["category"],
                    description=p.get("description", ""),
                    prompt=p["prompt"],
                )
            )
        return prompts

    def start_capture(self, prompt_entry: PromptEntry, context_id: str | None = None) -> RoutingTraceCollector:
        """Start capturing traces for a single prompt.

        Args:
            prompt_entry: The prompt entry to capture
            context_id: Optional context ID (generated if not provided)

        Returns:
            The RoutingTraceCollector instance for this capture
        """
        if context_id is None:
            context_id = f"{prompt_entry.id}_{uuid.uuid4().hex[:8]}"

        # Create trace file path
        trace_file = self.traces_dir / f"{prompt_entry.id}_{time.time_ns()}.parquet"

        # Create collector with prompt-specific ID
        collector = RoutingTraceCollector(
            output_dir=self.traces_dir,
            prompt_id=prompt_entry.id,
            token_category=prompt_entry.category,
        )
        collector.start(context_id=context_id, output_path=trace_file)

        self._current_collector = collector
        self._current_trace_file = trace_file
        return collector

    def stop_capture(self) -> Path | None:
        """Stop the current capture and return the trace file path.

        Returns:
            Path to the trace file, or None if no capture was active
        """
        if self._current_collector is None:
            return None

        self._current_collector.stop()
        trace_file = self._current_trace_file
        self._current_collector = None
        self._current_trace_file = None
        return trace_file

    def save_model_output(self, output: ModelOutput) -> Path:
        """Save model output to a machine-readable artifact.

        Args:
            output: The model output to save

        Returns:
            Path to the saved output file
        """
        output_file = self.outputs_dir / f"{output.prompt_id}_{int(time.time())}.json"

        data = {
            "prompt_id": output.prompt_id,
            "category": output.category,
            "context_id": output.context_id,
            "prompt_text": output.prompt_text,
            "generated_text": output.generated_text,
            "timestamp": output.timestamp,
            "metadata": output.metadata,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return output_file

    def capture_prompt(
        self,
        prompt_entry: PromptEntry,
        context_id: str | None = None,
        generated_text: str | None = None,
    ) -> CaptureResult:
        """Capture a single prompt with trace and output.

        This is a convenience method that handles the full capture lifecycle.
        If generate_fn was provided during initialization, it will be used
        to generate the model output. Otherwise, the caller should provide
        the generated_text or handle generation externally.

        Args:
            prompt_entry: The prompt entry to capture
            context_id: Optional context ID
            generated_text: Optional pre-generated text (uses generate_fn if not provided)

        Returns:
            CaptureResult with all capture information
        """
        # Start capture
        collector = self.start_capture(prompt_entry, context_id)
        context_id = collector.context_id

        # Generate or use provided text
        if generated_text is None and self.generate_fn is not None:
            generated_text = self.generate_fn(prompt_entry.prompt)
        elif generated_text is None:
            generated_text = ""

        # Stop capture
        trace_file = self.stop_capture()

        # Create and save model output
        model_output = ModelOutput(
            prompt_id=prompt_entry.id,
            category=prompt_entry.category,
            context_id=context_id,
            prompt_text=prompt_entry.prompt,
            generated_text=generated_text,
        )
        self.save_model_output(model_output)

        result = CaptureResult(
            prompt_entry=prompt_entry,
            model_output=model_output,
            trace_file=trace_file,
        )
        self.results.append(result)
        return result

    def capture_all(
        self,
        generate_fn: Callable[[str], str] | None = None,
        progress_callback: Callable[[int, int, PromptEntry], None] | None = None,
    ) -> list[CaptureResult]:
        """Capture all prompts in the suite.

        Args:
            generate_fn: Optional generation function (overrides the one from __init__)
            progress_callback: Optional callback(current, total, prompt_entry)

        Returns:
            List of CaptureResult for all prompts
        """
        prompts = self.load_prompt_suite()
        results = []

        gen_fn = generate_fn or self.generate_fn

        for i, prompt_entry in enumerate(prompts):
            if progress_callback:
                progress_callback(i, len(prompts), prompt_entry)

            # Generate text if function provided
            generated_text = None
            if gen_fn is not None:
                generated_text = gen_fn(prompt_entry.prompt)

            result = self.capture_prompt(prompt_entry, generated_text=generated_text)
            results.append(result)

        self.results = results
        return results

    def aggregate_traces(self, output_file: Path | None = None) -> Path:
        """Aggregate all captured traces into a single parquet file.

        Args:
            output_file: Optional output path (defaults to traces_dir/live_capture.parquet)

        Returns:
            Path to the aggregated trace file
        """
        if output_file is None:
            output_file = self.traces_dir / "live_capture.parquet"
        else:
            output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Collect all trace files
        all_records: list[dict] = []

        for result in self.results:
            if result.trace_file is None or not result.trace_file.exists():
                continue

            # Read the parquet file
            table = pq.read_table(result.trace_file)

            # Add prompt metadata to each record
            for batch in table.to_batches():
                for row in batch.to_pylist():
                    row["prompt_id"] = result.prompt_entry.id
                    row["category"] = result.prompt_entry.category
                    all_records.append(row)

        # Also scan for any other trace files in the traces directory
        # that might have been captured externally
        for trace_file in self.traces_dir.glob("*.parquet"):
            if trace_file.name == output_file.name:
                continue

            # Skip files already in results
            if any(r.trace_file == trace_file for r in self.results):
                continue

            try:
                table = pq.read_table(trace_file)
                for batch in table.to_batches():
                    for row in batch.to_pylist():
                        all_records.append(row)
            except Exception:
                # Skip files that can't be read
                continue

        if not all_records:
            # Create empty table with correct schema
            schema = pa.schema(
                [
                    ("token_id", pa.int64()),
                    ("context_id", pa.string()),
                    ("layer_id", pa.int64()),
                    ("token_position", pa.int64()),
                    ("expert_ids", pa.list_(pa.int64())),
                    ("expert_weights", pa.list_(pa.float64())),
                    ("token_text", pa.string()),
                    ("timestamp", pa.int64()),
                    ("token_category", pa.string()),
                    ("prompt_id", pa.string()),
                    ("category", pa.string()),
                ]
            )
            empty_table = pa.Table.from_pylist([], schema=schema)
            pq.write_table(empty_table, output_file, compression="zstd")
            return output_file

        # Write combined table
        table = pa.Table.from_pylist(all_records)
        pq.write_table(table, output_file, compression="zstd")

        return output_file

    def get_outputs_artifact(self) -> list[dict]:
        """Get all model outputs as a list of dictionaries.

        Returns:
            List of model output dictionaries
        """
        outputs = []
        for output_file in self.outputs_dir.glob("*.json"):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    outputs.append(json.load(f))
            except Exception:
                continue
        return outputs


def load_prompt_suite(path: Path | str) -> list[PromptEntry]:
    """Load a prompt suite from a JSON file.

    Args:
        path: Path to the prompt suite JSON file

    Returns:
        List of PromptEntry objects
    """
    capture = PromptSuiteCapture(output_dir=Path("/tmp"), prompt_suite_path=Path(path))
    return capture.load_prompt_suite()


def aggregate_trace_files(
    trace_files: list[Path],
    output_file: Path,
    prompt_metadata: dict[str, dict] | None = None,
) -> Path:
    """Aggregate multiple trace files into a single parquet file.

    Args:
        trace_files: List of trace file paths to aggregate
        output_file: Output path for the aggregated trace
        prompt_metadata: Optional dict mapping context_id to metadata dict

    Returns:
        Path to the aggregated trace file
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    prompt_metadata = prompt_metadata or {}

    for trace_file in trace_files:
        if not trace_file.exists():
            continue

        try:
            table = pq.read_table(trace_file)
            for batch in table.to_batches():
                for row in batch.to_pylist():
                    # Add metadata if available for this context_id
                    ctx_id = row.get("context_id", "")
                    if ctx_id in prompt_metadata:
                        row.update(prompt_metadata[ctx_id])
                    all_records.append(row)
        except Exception:
            continue

    if not all_records:
        schema = pa.schema(
            [
                ("token_id", pa.int64()),
                ("context_id", pa.string()),
                ("layer_id", pa.int64()),
                ("token_position", pa.int64()),
                ("expert_ids", pa.list_(pa.int64())),
                ("expert_weights", pa.list_(pa.float64())),
                ("token_text", pa.string()),
                ("timestamp", pa.int64()),
                ("token_category", pa.string()),
            ]
        )
        empty_table = pa.Table.from_pylist([], schema=schema)
        pq.write_table(empty_table, output_file, compression="zstd")
        return output_file

    table = pa.Table.from_pylist(all_records)
    pq.write_table(table, output_file, compression="zstd")
    return output_file
