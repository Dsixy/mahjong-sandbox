#!/usr/bin/env bash
# Run after the experiment-local runtime has been prepared on a CPU worker.
set -euo pipefail
policy_root="${1:?usage: run_policy_gpu.sh EXPERIMENT_ROOT}"
policy_python="$policy_root/runtime/bin/python"
cd -- "$policy_root/source"
mkdir -p "$policy_root/output" "$policy_root/cache/numba" "$policy_root/cache/matplotlib" "$policy_root/cache/cuda"
export POLICY_OUTPUT="$policy_root/output"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR="$policy_root/cache/numba"
export MPLCONFIGDIR="$policy_root/cache/matplotlib"
export CUDA_CACHE_PATH="$policy_root/cache/cuda"
export XDG_CACHE_HOME="$policy_root/cache"
exec > >(tee -a "$policy_root/output/worker.log") 2>&1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
"$policy_python" - <<'PY'
from datetime import datetime, timezone
import json, os
from pathlib import Path
import torch
from mahjong_sandbox.rl.checkpoint import source_fingerprint
if not torch.cuda.is_available():
    raise RuntimeError('GPU job has no usable CUDA; refusing a CPU fallback')
a = torch.randn(512, 512, device='cuda', requires_grad=True)
loss = (a @ a.T).square().mean()
loss.backward()
torch.cuda.synchronize()
if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(a.grad).all()):
    raise FloatingPointError('CUDA matmul/backward failed')
proof = {'utc': datetime.now(timezone.utc).isoformat(), 'torch': torch.__version__,
    'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
    'memory_bytes': torch.cuda.get_device_properties(0).total_memory,
    'cuda_matmul_backward': True, 'source_sha256': source_fingerprint(),
    'visible_cpus': len(os.sched_getaffinity(0))}
path = Path(os.environ['POLICY_OUTPUT']) / 'gpu-runtime.json'
path.write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps({'event': 'gpu_runtime_verified', **proof}), flush=True)
PY
exec "$policy_python" -m mahjong_sandbox.rl.longrun \
  --config configs/rl/policy-longrun.json \
  --output "$policy_root/output" \
  --initial models/sichuan-ppo-demo/best.zip \
  --additional-steps "${POLICY_ADDITIONAL_STEPS:-10000000}" \
  --eval-blocks 100
