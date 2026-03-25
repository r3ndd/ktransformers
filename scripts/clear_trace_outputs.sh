#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
TRACES_DIR="${ROOT_DIR}/data/traces"

if [ ! -d "${TRACES_DIR}" ]; then
  echo "Trace directory not found: ${TRACES_DIR}"
  exit 1
fi

shopt -s nullglob
files=("${TRACES_DIR}"/output_*.json)

if [ ${#files[@]} -eq 0 ]; then
  echo "No output_*.json files found in ${TRACES_DIR}"
  exit 0
fi

for file in "${files[@]}"; do
  : > "${file}"
done

echo "Cleared ${#files[@]} output JSON file(s) in ${TRACES_DIR}"
