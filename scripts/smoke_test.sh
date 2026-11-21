#!/usr/bin/env bash
# CPU-only end-to-end check with a tiny random OLMo-2 model and a scripted agent. ~3 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
# Keep this check CPU-only even when a GPU is visible but unavailable (e.g. on PACE).
export CUDA_VISIBLE_DEVICES=""
# Tiny models run faster with a small thread pool on shared many-core nodes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
python -m pytest -q tests
python scripts/01_build_data.py --n-test 800 --n-dev 240
python tests/make_tiny_model.py
M=artifacts/tiny_model
for m in oracle unedited prompting; do python scripts/02_run_method.py --model $M --method $m --rule chlorine --part dev_select --limit 6; done
for a in 0 1; do python scripts/03_wsp_edit.py --model $M --rule chlorine --budget 0.001 --attempt $a --mock-agent; done
python scripts/05_train_sft.py --model $M --rule chlorine --epochs 1
python scripts/06_rome.py --model $M --rule chlorine --cov-texts 200
python scripts/04_select_freeze.py --model $M --rules chlorine
for m in oracle unedited prompting wsp random_edit sft_lora; do
  python scripts/02_run_method.py --model $M --method $m --rule chlorine --part test --limit 30
done
python scripts/07_side_effects.py --model $M --rules chlorine --n-if 20
python scripts/08_analyze.py --n-boot 1000
echo "SMOKE TEST OK (tiny random model: scores are meaningless; remove artifacts/ and results/ before the real run)"
