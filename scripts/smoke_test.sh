#!/usr/bin/env bash
# CPU-only end-to-end check with a tiny random OLMo-2 model and a scripted agent. ~3 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
export WSPMOL_RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/wspmol-smoke.XXXXXXXX")
export WSPMOL_DATA_DIR="$WSPMOL_RUN_DIR/data"
echo "Smoke outputs: $WSPMOL_RUN_DIR"
# Keep this check CPU-only even when a GPU is visible but unavailable (e.g. on PACE).
export CUDA_VISIBLE_DEVICES=""
# Tiny models run faster with a small thread pool on shared many-core nodes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
python -m pytest -q tests
python scripts/01_build_data.py --n-test 30 --n-dev 24
python tests/make_tiny_model.py
M="$WSPMOL_RUN_DIR/artifacts/tiny_model"
for m in oracle unedited prompting; do python scripts/02_run_method.py --model "$M" --method $m --rule chlorine --part dev_select --limit 6; done
for rule in chlorine mass; do
  for budget in 0.001 0.0001; do
    for attempt in 0 1; do
      python scripts/03_wsp_edit.py --model "$M" --rule "$rule" --budget "$budget" --attempt "$attempt" --mock-agent
    done
  done
done
python scripts/05_train_sft.py --model "$M" --rule chlorine --epochs 1
python scripts/06_rome.py --model "$M" --rule chlorine --cov-texts 200
python scripts/04_select_freeze.py --model "$M" --mock-agent
python - "$WSPMOL_RUN_DIR/artifacts/edits/FROZEN.json" <<'PYTHON'
import json
import sys
from pathlib import Path

frozen = json.loads(Path(sys.argv[1]).read_text())
for rule in ("chlorine", "mass"):
    row = frozen[rule]
    assert row["primary"] and not row["missing"]
    assert len(row["attempts"]) == 4 and all(a["usable"] for a in row["attempts"])
PYTHON
for m in oracle unedited prompting wsp random_edit sft_lora; do
  python scripts/02_run_method.py --model "$M" --method $m --rule chlorine --part test --limit 30
done
for rule in chlorine mass; do
  plan=$(python -m wspmol.wsp.frozen "$WSPMOL_RUN_DIR/artifacts/edits/FROZEN.json" "$rule" "$M")
  for name in $plan; do
    [[ "$name" == primary ]] && continue
    python scripts/02_run_method.py --model "$M" --rule "$rule" --method wsp_attempt --part test --limit 2 \
      --tag "wsp_b${name##*_b}" --edit "$WSPMOL_RUN_DIR/artifacts/edits/$name.pt"
  done
done
python scripts/07_side_effects.py --model "$M" --rules chlorine --n-if 20
python scripts/08_analyze.py --n-boot 1000
echo "SMOKE TEST OK (tiny random model and scripted agent)"
