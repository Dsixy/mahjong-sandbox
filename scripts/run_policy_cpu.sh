#!/usr/bin/env bash
# RUN_ROOT supplies config.json, initial.zip/.json, source/ and a prepared runtime/.
set -euo pipefail
policy_run="${1:?usage: run_policy_cpu.sh RUN_ROOT}"
policy_run="$(cd -- "$policy_run" && pwd -P)"
policy_python="$policy_run/runtime/bin/python"
test -x "$policy_python"
test -f "$policy_run/config.json"
test -f "$policy_run/initial.zip"
cd -- "$policy_run/source"
mkdir -p "$policy_run/output" "$policy_run/cache/numba" "$policy_run/cache/matplotlib"
export POLICY_CPU_RUN="$policy_run"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR="$policy_run/cache/numba"
export MPLCONFIGDIR="$policy_run/cache/matplotlib"
export XDG_CACHE_HOME="$policy_run/cache"
exec > >(tee -a "$policy_run/output/worker.log") 2>&1
"$policy_python" - <<'PY'
from datetime import datetime, timezone
import json, os
from pathlib import Path
import torch
from mahjong_sandbox.rl.checkpoint import load_checkpoint, source_fingerprint
from mahjong_sandbox.rl.train import TrainConfig
root = Path(os.environ['POLICY_CPU_RUN'])
config = TrainConfig(**json.loads((root / 'config.json').read_text()))
if config.device != 'cpu':
    raise ValueError('CPU launch requires device=cpu')
torch.set_num_threads(config.torch_threads)
model, metadata = load_checkpoint(root / 'initial.zip', device='cpu', expected_rules=config.rules)
if config.hints != metadata['hints']:
    raise ValueError('initial checkpoint uses different features')
if not all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()):
    raise FloatingPointError('initial model contains nonfinite parameters')
optimizer_tensors = [value for state in model.policy.optimizer.state.values()
                     for value in state.values() if torch.is_tensor(value)]
if any(value.device.type != 'cpu' for value in optimizer_tensors):
    raise RuntimeError('optimizer was not restored on CPU')
a = torch.randn(256, 256, requires_grad=True)
loss = (a @ a.T).square().mean()
loss.backward()
if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(a.grad).all()):
    raise FloatingPointError('CPU matmul/backward failed')
proof = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'torch': torch.__version__,
    'device': str(model.device), 'visible_cuda_devices': torch.cuda.device_count(),
    'initial_timesteps': model.num_timesteps, 'optimizer_epochs': model._n_updates,
    'optimizer_tensor_count': len(optimizer_tensors), 'initial_sha256': metadata['model_sha256'],
    'source_sha256': source_fingerprint(), 'cpu_matmul_backward': True}
(root / 'output/cpu-runtime.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps({'event': 'cpu_runtime_verified', **proof}), flush=True)
PY
policy_steps="${POLICY_ADDITIONAL_STEPS:-1000000}"
exec "$policy_python" -m mahjong_sandbox.rl.longrun \
  --config "$policy_run/config.json" \
  --output "$policy_run/output" \
  --initial "$policy_run/initial.zip" \
  --additional-steps "$policy_steps" \
  --eval-blocks "${POLICY_EVAL_BLOCKS:-100}"
