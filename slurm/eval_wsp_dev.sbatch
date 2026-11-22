#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
unset PYTHONHOME PYTHONPATH
export HF_HOME="$SLURM_SUBMIT_DIR/.hf_cache" HF_HUB_OFFLINE=1
P=$HOME/scratch/conda-envs/wspmol/bin/python
for f in ${WSPMOL_RUN_DIR:-.}/artifacts/edits/wsp_${RULE}_b*_a*.pt; do
  n=$(basename "$f" .pt)
  echo "=== $n"
  "$P" scripts/02_run_method.py --model "${MODEL:-allenai/OLMo-2-0425-1B-Instruct}" --method wsp_attempt \
    --edit "$f" --tag "$n" --rule "$RULE" --part dev_select --batch-size ${BS:-32}
done
