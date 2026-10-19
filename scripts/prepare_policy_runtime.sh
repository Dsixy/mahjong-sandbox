#!/usr/bin/env bash
# One-shot CPU RJob: install into this experiment only, never a shared base env.
set -euo pipefail
policy_root="${1:?usage: prepare_policy_runtime.sh EXPERIMENT_ROOT}"
cd -- "$policy_root/source"
mkdir -p "$policy_root/cache/pip" "$policy_root/preparation"
export PIP_CACHE_DIR="$policy_root/cache/pip"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PIP_CONFIG_FILE=/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PIP_EXTRA_INDEX_URL
exec > >(tee -a "$policy_root/preparation/prepare.log") 2>&1
policy_base_python=''
for policy_candidate in python3 python /opt/conda/bin/python /root/miniconda3/bin/python; do
  if command -v "$policy_candidate" >/dev/null && "$policy_candidate" -c 'import sys; assert sys.version_info >= (3,11)' 2>/dev/null; then
    policy_base_python="$policy_candidate"
    break
  fi
done
test -n "$policy_base_python"
"$policy_base_python" -m venv "$policy_root/runtime"
policy_python="$policy_root/runtime/bin/python"
policy_pip_options=(--index-url "${POLICY_PYPI_INDEX_URL:?set a CPU-accessible package index}" --find-links "$policy_root/wheels")
if [[ -n "${POLICY_PYPI_TRUSTED_HOST:-}" ]]; then
  policy_pip_options+=(--trusted-host "$POLICY_PYPI_TRUSTED_HOST")
fi
"$policy_python" -m pip install "${policy_pip_options[@]}" \
  'setuptools>=68' wheel 'torch==2.8.0' 'numpy==2.3.5' 'numba==0.68.0' 'llvmlite==0.50.0' \
  'gymnasium==1.4.0' 'stable-baselines3==2.9.0' 'sb3-contrib==2.9.0'
"$policy_python" -m pip install --no-deps --no-build-isolation -e .
"$policy_python" -m pip check
"$policy_python" -m pip freeze > "$policy_root/preparation/requirements.lock.txt"
"$policy_python" - <<'PY'
import json, platform
from pathlib import Path
import torch
from mahjong_sandbox.rl.checkpoint import load_checkpoint
model, metadata = load_checkpoint('models/sichuan-ppo-demo/best.zip', device='cpu')
if torch.version.cuda is None:
    raise RuntimeError('preparation produced CPU-only Torch')
print(json.dumps({'event': 'runtime_prepared', 'python': platform.python_version(),
    'torch': torch.__version__, 'cuda_build': torch.version.cuda,
    'initial_timesteps': metadata['timesteps'], 'model_device': str(model.device)}), flush=True)
PY
"$policy_python" -m unittest tests.test_rl tests.test_selfplay -q
touch "$policy_root/preparation/ready"
