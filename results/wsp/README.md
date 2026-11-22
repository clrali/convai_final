# WSP edit results (dev_select), Nov 21 2026

GPT-6.1 Sol edited OLMo 2 weights using forward probes only (§4 of the proposal). This folder holds every
edit attempt plus same-run baselines, scored on `dev_select` (120 scenarios per condition per rule).
**No edit is frozen and the test set has not been opened.**

**Headline (13B):** Sol produced a usable edit in all 8 attempts, but none changes behaviour. C2 (new-session)
turn accuracy stays at the unedited level (chlorine 27–28% vs 27.5%; mass 37–41% vs 38.6%), while saved-rule
prompting reaches 73.3% / 50.3%. Clara's SFT (LoRA) reaches 99.4% / 92.5%, and ROME also has no effect
(see `results/report_dev_select.md`). Under H1, WSP fails. Under H2, WSP passes, but only because the edits
change almost nothing.

## Setup

| | |
|---|---|
| Target | `allenai/OLMo-2-1124-13B-Instruct` (main), `allenai/OLMo-2-0425-1B-Instruct` (first run) |
| Editor | `gpt-6.1-sol`, Responses API, ≤25 probe batches, ≤80 steps, seed 0 |
| Attempts | 2 rules × 2 budgets (0.1%, 0.01%) × 2 attempts = 8 per model |
| Data | code at `bf9777f` (project IDs `chlorine` / `mass`), ChEBI library of 17,705 molecules, `dev_train` sha256 `af3f13c0…` |
| Hardware | PACE-ICE, one H100 per job, bf16 |
| Pilot | XOR repair passed (accuracy 50% → 100%, 1 weight changed, $0.004): `pilot_small_network.json` |

## OLMo 2 13B: dev_select

Turn accuracy / dialogue success (%). "over" is C3 over-application; "post" is C4 success after the change.

### chlorine

| method | C1 | C2 | C3 | over | C4 | post | mean succ |
|---|---|---|---|---|---|---|---|
| oracle | 100/100 | 100/100 | 100/100 | 0.0 | 100/100 | 100 | 100 |
| unedited | 71.4/25.8 | 27.5/0.0 | 86.9/67.5 | 13.3 | 80.0/54.2 | 65.8 | 36.9 |
| prompting | 77.5/38.3 | **73.3/30.8** | 86.7/65.8 | 12.5 | 83.1/60.8 | 68.3 | 49.0 |
| WSP 0.01% a0 | 70.6/27.5 | 27.2/0.0 | 88.3/69.2 | 10.8 | 80.8/52.5 | 60.8 | 37.3 |
| WSP 0.01% a1 | 71.1/26.7 | 27.5/0.0 | 87.5/68.3 | 12.9 | 81.1/56.7 | 65.8 | 37.9 |
| WSP 0.1% a0 | 71.4/26.7 | 28.3/0.0 | 85.8/64.2 | 13.8 | 78.6/51.7 | 61.7 | 35.6 |
| WSP 0.1% a1 | 70.6/25.8 | 27.8/0.0 | 86.4/65.0 | 14.2 | 79.7/51.7 | 60.8 | 35.6 |

### mass

| method | C1 | C2 | C3 | over | C4 | post | mean succ |
|---|---|---|---|---|---|---|---|
| oracle | 100/100 | 100/100 | 100/100 | 0.0 | 100/100 | 100 | 100 |
| unedited | 50.0/6.7 | 38.6/2.5 | 88.3/70.0 | 10.8 | 80.6/52.5 | 81.7 | 32.9 |
| prompting | 55.0/10.8 | **50.3/8.3** | 88.1/68.3 | 11.2 | 82.8/61.7 | 79.2 | 37.3 |
| WSP 0.01% a0 | 52.8/14.2 | 41.4/3.3 | 84.7/60.8 | 16.2 | 77.2/48.3 | 72.5 | 31.7 |
| WSP 0.01% a1 | 51.4/8.3 | 39.4/3.3 | 86.4/65.0 | 14.2 | 80.8/51.7 | 82.5 | 32.1 |
| WSP 0.1% a0 | 51.9/7.5 | 40.6/2.5 | 85.0/63.3 | 15.4 | 79.2/47.5 | 80.0 | 30.2 |
| WSP 0.1% a1 | 46.4/1.7 | 36.7/0.0 | 89.4/70.8 | 7.1 | 80.0/51.7 | 77.5 | 31.0 |

Prompting dev turn accuracy (the proposal's 70% risk check): chlorine **80.1%** (passes), mass **69.0%**
(just below). Mass is harder for the model even with the rule in context (C1 50%).

## OLMo 2 1B: dev_select (floor effect)

The 1B model is near chance even when the rule is in context (C1 20–27% turn accuracy, about 0% dialogue
success), and prompting ≈ unedited (dev turn accuracy 35.3% / 33.5%, far below 70%). WSP edits on 1B moved
C2 by at most a few points, and the mass 0.01% a0 edit raised C3 over-application from 24.6% to 41.7%. These
runs are kept for completeness but don't test the hypotheses. Full tables: `python scripts/summarize_dev.py results/wsp/olmo1b/dev_select`.

## Cost

| | attempts | API cost |
|---|---|---|
| 13B | 8 | $10.90 ($1.00–$2.09 each) |
| 1B | 8 | $10.59 ($1.03–$2.24 each) |
| pilot | 1 | $0.004 |

Each attempt used all 25 probe batches and 34–54 agent steps, and took about 3–7 minutes on an H100.

## Caveats

- In-run numbers in the agent notes (`edits/*.json` → `notes`) come from about 12 dialogues per condition on
  `dev_train`. They overstate gains that disappear on `dev_select`.
- `wsp_mass_b0.0001_a0` (13B) was submitted at a scale the agent did not evaluate (it interpolated between
  tested scales).
- No paired bootstrap has been run on these rows yet. Predictions are included for that.
- Clara's SFT trains on `dev_train` and is scored on `dev_select`, which share templates. The test set uses
  disjoint templates and molecules.

## Files

```
pilot_small_network.json          XOR pilot (Sol, one-scalar repair)
olmo13b/edits/wsp_<rule>_b<budget>_a<n>.json    edit spec, digest, probe log, full transcript, cost
olmo13b/dev_select/<rule>/<method>/             summary.json, predictions.jsonl, log.txt
olmo1b/...                                      same layout for the 1B run
```

The edit weight files (`*.pt`; 164 MB each at 13B 0.1%) are too big for GitHub. They are on PACE under
`~aonaindia3/scratch/convai_final/{artifacts,runs/olmo13b/artifacts}/edits/`. Their sha256 digests are in
the JSON files.

## Reproduce

```bash
# edits (one job per rule x budget; RULES / WSP_BUDGETS override run_all.sh defaults)
sbatch -t 02:00:00 --export=ALL,STAGE=wsp,RULES=chlorine,WSP_BUDGETS=0.001,BS=16,\
MODEL=allenai/OLMo-2-1124-13B-Instruct,WSPMOL_RUN_DIR=$PWD/runs/olmo13b slurm/stage.sbatch
# baselines, then score every attempt on dev_select without freezing
sbatch -t 01:00:00 --export=ALL,STAGE=baselines_dev,BS=16,MODEL=...,WSPMOL_RUN_DIR=... slurm/stage.sbatch
sbatch --export=ALL,RULE=chlorine,BS=16,MODEL=...,WSPMOL_RUN_DIR=... slurm/eval_wsp_dev.sbatch
python scripts/summarize_dev.py runs/olmo13b/results/dev_select
python scripts/summarize_edits.py runs/olmo13b/artifacts/edits
```

On Nov 21, the GPUs on `atl1-1-03-011-28-0` and one on `atl1-1-03-010-25-0` failed the CUDA allocation check.
Use `--exclude` for those nodes if a job fails there.
