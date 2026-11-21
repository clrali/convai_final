# Can Weight-Space Programming Preserve Molecular Design Constraints Across Dialogue Turns?

Implementation of the experiment in the ECE 6256 / CS 6604 proposal (De Onaindia, Patil, Saha, Li).
An LLM agent (GPT-6.1 Sol) edits OLMo 2 1B Instruct's weights **using forward probes only**, so the
assistant keeps applying a project rule (mass ≤ 250 Da, or no chlorine) across sessions. It is
compared against an unedited model, saved-rule prompting, SFT (LoRA), ROME, a random matched edit
and an RDKit oracle.

## Layout

| Paper section | Code |
|---|---|
| §3 dialogue conditions C1–C4, pass criterion | `wspmol/scenarios.py`, `wspmol/rules.py`, `wspmol/prompts.py` |
| §4 WSP: forward probes, reversible edits, 0.1% / 0.01% budgets, ≤25 probe batches | `wspmol/wsp/probes.py`, `wspmol/wsp/agent.py`, `wspmol/edits.py`, `scripts/03_wsp_edit.py` |
| §4 select primary 0.1% edit on dev, freeze, report both attempts | `scripts/04_select_freeze.py` (writes `artifacts/edits/FROZEN.json`) |
| §4 size- and norm-matched random edit | `edits.random_matched_edit`, method `random_edit` |
| §5 test set: 800 scenarios/rule × C1–C4, disjoint dev/test molecules and templates, shuffled order, half ID+table lists | `scripts/01_build_data.py` |
| §5 ChEBI + RDKit labels | `scripts/00_fetch_chebi.py`, `wspmol/molecules.py` |
| §5 baselines | `wspmol/methods/` (`sft.py`, `rome.py`, `registry.py`), oracle in `runner.run_oracle` |
| §5 scoring: turn acc, dialogue success, C4 post-change, C3 over-application | `wspmol/scoring.py` |
| §5 paired scenario bootstrap, H1/H2 margins, power (n≈800 → ±4.9 pts) | `wspmol/stats.py`, `scripts/08_analyze.py` |
| §5 100-turn manual audit | `results/audit_test.csv` (fill in `human_label`) |
| §5 rollback + 200-item instruction-following slice | `wspmol/side_effects.py`, `scripts/07_side_effects.py` |
| §6 Sol pilot on a small network; prompting < 70% check | `scripts/pilot_small_network.py`, `scripts/check_prompting_dev.py` |

## Setup on PACE-ICE

```bash
cd /home/hice1/cli725/scratch/conv_ai/CAI_FinalProject
module load anaconda3
conda create -n wspmol python=3.11 -y && conda activate wspmol
pip install -r requirements.txt
export OPENAI_API_KEY=...          # WSP agent + pilot only
```

Check everything works without a GPU or API key first (tiny random model plus a scripted agent, about 3 minutes):

```bash
bash scripts/smoke_test.sh && rm -rf artifacts results
```

## Running

Stages, matching the milestones (each one can be submitted as a job):

```bash
sbatch --export=ALL,STAGE=data slurm/stage.sbatch            # runs for ~5 min
sbatch --export=ALL,STAGE=pilot slurm/stage.sbatch
sbatch --export=ALL,STAGE=baselines_dev slurm/stage.sbatch   # Nov 21: baselines + throughput (runs for ~7 min)
sbatch --export=ALL,STAGE=wsp slurm/stage.sbatch             # 8 edit runs (needs API access from the node)
sbatch --export=ALL,STAGE=adapt slurm/stage.sbatch           # SFT (LoRA) + ROME
sbatch --export=ALL,STAGE=freeze slurm/stage.sbatch          # select + freeze + power check
sbatch --export=ALL,STAGE=test slurm/stage.sbatch            # opens the test set
sbatch --export=ALL,STAGE=side slurm/stage.sbatch
STAGE=analyze bash scripts/run_all.sh       # -> results/report_test.md
```

Slurm jobs invoke `$HOME/scratch/conda-envs/wspmol/bin/python` directly for every stage,
without relying on shell activation. They log the Python executable and PyTorch version before
starting and clear inherited `PYTHONHOME`/`PYTHONPATH` settings. If your environment is elsewhere,
set `export WSPMOL_ENV=/full/path/to/wspmol` before submitting; `--export=ALL,...` passes it to the job.
Environment setup failures stop the job immediately.

`02_run_method.py` refuses to run WSP on the test set until the edits are frozen. It also skips ROME
when ROME failed its factual rewrite check (`--force` overrides). The fallback in the plan is to
report that failure and finish the other comparisons.

Set `AGENT_MODEL` to the real API model ID for Sol. For the Astra fallback, pass
`--base-url` / `--agent-model` to `03_wsp_edit.py`. Token usage and estimated cost
($2/M input, $10/M output) are logged in `artifacts/edits/wsp_*.json`.

## Design choices the proposal leaves open (change them before freezing)

- **Closeness** is Morgan (r=2, 2048-bit) Tanimoto similarity to the starting molecule. It is shown
  in every candidate list along with MW and chlorine, following "we supply each candidate's properties".
- **Edited project IDs**: `Heron` (chlorine), `Juniper` (mass). C3 uses neutral control names from
  pools that differ between dev and test, assigned round-robin so they're balanced.
- **Turns per scenario**: C1/C2/C3 use three lists in shuffled order: *rel* (closest candidate breaks
  the rule), *none* (no candidate is eligible) and *comply*. C4 has one pre-change list, then the
  revision, then two lists whose answer must change. Revision: the mass limit goes up to 350 Da, or
  chlorine becomes allowed.
- **Rule store**: in C2 only prompting and the oracle get the saved rule. In C4 their store updates.
  WSP, SFT and ROME keep their adaptation in every condition.
- **WSP edit primitives**: sparse rank-1 (`scale·u[rows]v[cols]ᵀ` on the top-|u| rows × top-|v| cols)
  or explicit entries, built from probe vectors (activation captures, token unembeddings, vector
  algebra). Each `capture` / `next_token_logprobs` / `evaluate` call costs one of the 25 probe batches.
- **Primary edit selection**: highest mean full-dialogue success over C1–C4 on `dev_select`.
  The agent and SFT only see `dev_train`.
- **ROME**: layer `L/2-1` (layer 7 of 16), `mlp.down_proj`, covariance from wikitext-103 if it
  downloads, otherwise from dev dialogue text.
- **Instruction-following slice**: 200 verifiable synthetic instructions (case, length, keywords,
  bullets, JSON, …). IFEval itself isn't used.
- **Molecule library**: `wspmol/data/builtin_molecules.tsv` has 261 hand-checked molecules (95
  chlorinated). `00_fetch_chebi.py` expands it from ChEBI 3-star and adds ChEBI IDs. Molecules
  without an ID show an opaque `MOL:xxxxxx` ID in ID-mode lists.

`wspmol/wsp/mock_agent.py` is a scripted stand-in used only for pipeline tests. It is **not** WSP.
