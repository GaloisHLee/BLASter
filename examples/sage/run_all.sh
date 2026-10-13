#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-sage106}"

for example in "$SCRIPT_DIR"/[0-9][0-9]_*.sage; do
    echo "===== $(basename "$example") ====="
    conda run --no-capture-output -n "$CONDA_ENV" sage "$example"
done
