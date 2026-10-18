#!/usr/bin/env bash
# Run from the repository root, in a Python 3.10 Linux x86_64 environment.
set -euo pipefail
python -m pip install --require-hashes -r "${1:-ci/requirements.lock}"
python -m pip check
