#!/usr/bin/env bash
# Preserve every pytest failure, including exit 5 (no tests collected).
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" SONAR_CPU_CI=1 MPLBACKEND=Agg
export RUN_RENDERER_BASELINE_RUNTIME_SMOKES=0 RUN_CHUNK4_RUNTIME_SMOKES=0
export RUN_CHUNK5_RUNTIME_SMOKES=0 RUN_CHUNK4_SYNTHETIC_MATRIX=0
# Required packages must not disappear behind inherited importorskip calls.
python -c 'import torch, numpy, scipy, yaml, PIL, cv2, matplotlib, pytest, pytest_timeout, plyfile; assert torch.version.cuda is None, "CPU CI requires a CPU-only torch wheel"'
if [ "$#" -eq 0 ]; then set -- tests; fi
python -m pytest "$@" -q -ra --strict-markers --timeout=600 --timeout-method=thread -p no:cacheprovider
