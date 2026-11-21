#!/bin/bash
# Usage: sbatch --export=ALL,STAGE=baselines_dev slurm/stage.sbatch
# Stages: data pilot baselines_dev wsp adapt freeze test side analyze
#SBATCH -J wsp-mol
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1               
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
# Invoke the environment's Python directly; inherited Conda state and modules
# can leave a batch shell resolving `python` to the base installation.
export WSPMOL_ENV="${WSPMOL_ENV:-$HOME/scratch/conda-envs/wspmol}"
export WSPMOL_PYTHON="$WSPMOL_ENV/bin/python"
if [[ ! -x "$WSPMOL_PYTHON" ]]; then
  echo "Environment Python is not executable: $WSPMOL_PYTHON" >&2
  exit 1
fi
export PATH="$WSPMOL_ENV/bin:$PATH"
unset PYTHONHOME PYTHONPATH
"$WSPMOL_PYTHON" -u - <<'PYTHON'
import os
import socket
import sys

print(f"Python: {sys.executable}", flush=True)
print(f"Environment: {sys.prefix}", flush=True)
import torch
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"Host: {socket.gethostname()}; job: {os.getenv('SLURM_JOB_ID')}; "
      f"allocated GPUs: {os.getenv('SLURM_JOB_GPUS')}; "
      f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}", flush=True)
# is_available() alone can succeed even when CUDA cannot create a context.
# Check a real allocation before loading model weights.
try:
    probe = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
except RuntimeError as exc:
    raise SystemExit(f"CUDA allocation check failed on {socket.gethostname()}: {exc}") from exc
free, total = torch.cuda.mem_get_info()
print(f"CUDA allocation check passed: {torch.cuda.get_device_name(0)}; "
      f"{free / 2**30:.2f}/{total / 2**30:.2f} GiB free", flush=True)
PYTHON
export HF_HOME="$SLURM_SUBMIT_DIR/.hf_cache"
export STAGE="${STAGE:-all}"
exec bash scripts/run_all.sh "${MODEL:-allenai/OLMo-2-1124-13B-Instruct}"
