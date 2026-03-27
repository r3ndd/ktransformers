#!/usr/bin/env python3
"""Run a single-prompt sanity check via run_collection.py.

This wrapper executes run_collection for prompt id ``coding_001`` by default.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_COLLECTION = ROOT / "scripts" / "run_collection.py"
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity check wrapper around run_collection.py"
    )
    parser.add_argument(
        "--prompt-id",
        default="coding_001",
        help="Prompt id to run from the suite (default: coding_001)",
    )
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Continue on failures (passed through to run_collection.py)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not RUN_COLLECTION.is_file():
        raise FileNotFoundError(f"run_collection.py not found at: {RUN_COLLECTION}")

    cmd = [
        PYTHON_BIN,
        str(RUN_COLLECTION),
        "--prompt-id",
        args.prompt_id,
    ]
    if args.no_fail_fast:
        cmd.append("--no-fail-fast")

    print(f"[sanity_check] Running: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
