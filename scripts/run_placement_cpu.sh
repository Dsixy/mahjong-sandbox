#!/usr/bin/env bash
set -euo pipefail
placement_root="${1:?usage: run_placement_cpu.sh RUN_ROOT LABEL}"
placement_label="${2:?training label required}"
placement_python="$placement_root/runtime/bin/python"
cd -- "$placement_root/source"
mkdir -p "$placement_root/$placement_label" "$placement_root/cache/$placement_label"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR="$placement_root/cache/$placement_label"
exec "$placement_python" scripts/run_placement_ablation.py --root "$placement_root" \
  --worker-config "$placement_root/$placement_label.json" \
  --output "$placement_root/$placement_label" --initial "$placement_root/initial.zip"
