#!/usr/bin/env bash
# RUN_ROOT contains source/, runtime/, config.json and persistent output/.
set -euo pipefail
riichi_run="${1:?usage: run_riichi_cpu.sh RUN_ROOT}"
riichi_run="$(cd -- "$riichi_run" && pwd -P)"
riichi_python="$riichi_run/runtime/bin/python"
cd -- "$riichi_run/source"
mkdir -p "$riichi_run/output" "$riichi_run/cache/numba" "$riichi_run/cache/matplotlib"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR="$riichi_run/cache/numba"
export MPLCONFIGDIR="$riichi_run/cache/matplotlib"
export XDG_CACHE_HOME="$riichi_run/cache"
exec > >(tee -a "$riichi_run/output/worker.log") 2>&1
riichi_steps="$($riichi_python -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["variant"]=="riichi" and c["device"]=="cpu"; print(c["total_timesteps"])' "$riichi_run/config.json")"
exec "$riichi_python" -m mahjong_sandbox.rl.longrun \
  --config "$riichi_run/config.json" --output "$riichi_run/output" \
  --additional-steps "$riichi_steps" --eval-blocks "${RIICHI_EVAL_BLOCKS:-16}"
