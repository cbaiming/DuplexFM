#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${DUPLEXFM_PYTHON:?Set DUPLEXFM_PYTHON to existing environment python}"
ASSETS="${DUPLEXFM_ASSETS:?Set DUPLEXFM_ASSETS to existing assets}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
test -f "$ROOT/parallel_registration.json"
test -f "$ROOT/environment/ready.json"
test -f "$ROOT/environment/prepared_backup.json"
ARGS=(run --root "$ROOT" --assets "$ASSETS" --gpus "${DUPLEXFM_GPUS:-0,1}")
if [[ -n "${DUPLEXFM_PREVIOUS_STOPPED_AT:-}" ]]; then
    ARGS+=(--previous-stopped-at "$DUPLEXFM_PREVIOUS_STOPPED_AT")
fi
exec "$PY" -u "$ROOT/code/parallel_run.py" "${ARGS[@]}"
