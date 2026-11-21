#!/usr/bin/env bash
# Full experiment. Run stages one at a time on PACE-ICE (see slurm/), or all at once:
#   bash scripts/run_all.sh [MODEL]
# Requires OPENAI_API_KEY for the WSP agent (stage 3) and the pilot.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-allenai/OLMo-2-0425-1B-Instruct}
AGENT=${AGENT_MODEL:-gpt-6.1-sol}
API=${AGENT_API:-responses}
EDITS="${WSPMOL_RUN_DIR:-.}/artifacts/edits"
BS=${BS:-32}
SFT_BS=${SFT_BS:-1}
SFT_GRAD_ACCUM=${SFT_GRAD_ACCUM:-8}
RULES="chlorine mass"
STAGE=${STAGE:-all}
PYTHON=${WSPMOL_PYTHON:-python}
run() { [[ "$STAGE" == all || "$STAGE" == "$1" ]]; }

if run data; then
  [[ -f wspmol/data/chebi_library.tsv ]] || "$PYTHON" scripts/00_fetch_chebi.py || echo "ChEBI fetch failed; using built-in library"
  "$PYTHON" scripts/01_build_data.py --n-test 800 --n-dev 240
fi

if run pilot; then
  "$PYTHON" scripts/pilot_small_network.py --task xor --agent-model "$AGENT" --agent-api "$API"
fi

if run baselines_dev; then   # Nov 19 milestone: baselines + throughput check
  for r in $RULES; do for m in oracle unedited prompting; do
    "$PYTHON" scripts/02_run_method.py --model "$MODEL" --method $m --rule $r --part dev_select --batch-size $BS
  done; done
  "$PYTHON" scripts/check_prompting_dev.py
fi

if run wsp; then             # 2 rules x 2 budgets x 2 attempts = 8 edit runs
  for r in $RULES; do for b in 0.001 0.0001; do for att in 0 1; do
    "$PYTHON" scripts/03_wsp_edit.py --resume --model "$MODEL" --rule $r --budget $b --attempt $att \
      --agent-model "$AGENT" --agent-api "$API" --batch-size $BS
  done; done; done
fi

if run adapt; then           # SFT (LoRA) + ROME
  for r in $RULES; do
    "$PYTHON" scripts/05_train_sft.py --model "$MODEL" --rule $r \
      --batch-size "$SFT_BS" --grad-accum "$SFT_GRAD_ACCUM"
    "$PYTHON" scripts/06_rome.py --model "$MODEL" --rule $r
    "$PYTHON" scripts/02_run_method.py --model "$MODEL" --method sft_lora --rule $r --part dev_select --batch-size $BS
  done
fi

if run freeze; then
  "$PYTHON" scripts/04_select_freeze.py --model "$MODEL" --batch-size $BS
fi

if run test; then            # opens the test set
  for r in $RULES; do
    plan=$("$PYTHON" -m wspmol.wsp.frozen "$EDITS/FROZEN.json" "$r" "$MODEL")
    for m in oracle unedited prompting sft_lora rome; do
      "$PYTHON" scripts/02_run_method.py --model "$MODEL" --method "$m" --rule "$r" --part test \
        --skip-failed-rome --batch-size "$BS"
    done
    for name in $plan; do
      if [[ "$name" == primary ]]; then
        for m in wsp random_edit; do
          "$PYTHON" scripts/02_run_method.py --model "$MODEL" --method "$m" --rule "$r" --part test --batch-size "$BS"
        done
      else
        "$PYTHON" scripts/02_run_method.py --model "$MODEL" --method wsp_attempt --edit "$EDITS/$name.pt" \
          --tag "wsp_b${name##*_b}" --rule "$r" --part test --batch-size "$BS"
      fi
    done
  done
fi

if run side; then
  "$PYTHON" scripts/07_side_effects.py --model "$MODEL" --batch-size $BS
fi

if run analyze; then
  "$PYTHON" scripts/08_analyze.py --part test
fi
