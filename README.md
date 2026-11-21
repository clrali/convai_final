# Can Weight-Space Programming Preserve Molecular Design Constraints Across Dialogue Turns?

Implementation of the experiment in the ECE 6256 / CS 6604 proposal (De Onaindia, Patil, Saha, Li).
The experiment tests whether GPT-6.1 Sol can edit OLMo 2 1B Instruct's weights **using forward probes only**
to carry a project rule (mass ≤ 250 Da, or no chlorine) across sessions. It is
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
export WSPMOL_ENV="$CONDA_PREFIX"
export OPENAI_API_KEY=...          # WSP agent + pilot only
```

The offline smoke test uses a tiny random OLMo 2, runs the eight scripted edit attempts and prints the temporary directory containing its generated datasets, edits, predictions and report.

```bash
bash scripts/smoke_test.sh
```

## Running the GPT editor

With the dependencies installed and `OPENAI_API_KEY` set, run one development attempt from the repository root. The checked-in development files are ready to use.

```bash
python scripts/03_wsp_edit.py --rule chlorine --budget 0.001 --attempt 0
```

The editor receives the project ID, the rule and labelled `dev_train` examples. GPT can inspect selected weight entries, compare activation vectors and token scores, propose a sparse edit, and observe its effects on development dialogues. A replacement must fit the entry budget and change at least one stored weight at the target precision. Each activation, log-probability or dialogue evaluation request consumes one of 25 logical probe batches. The runtime may split that batch into smaller forward passes to fit memory. Selection uses a separate `dev_select` split.

The default agent is `gpt-6.1-sol` through the Responses API. Set `AGENT_MODEL` to choose another editor. For a compatible endpoint, set `OPENAI_BASE_URL` or pass `--base-url`. For a chat-completions-only endpoint, use `--agent-api chat` (or `AGENT_API=chat` with the stage launcher). [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-6.1-sol)

Each attempt writes its transcript, probe count, token usage and stop reason to `artifacts/edits/wsp_*.json`. A usable edit also has a `.pt` file and SHA-256 digest. The cost field uses the proposal's Sol rates of $2/M input and $10/M output tokens. The WSP stage resumes completed attempts with matching data and settings. An API failure records the partial transcript and exits the stage so it can be retried.

The pilot stage asks the editor to repair a published nine-parameter XOR network with one scalar change. It scores all four inputs and records the before/after accuracy and number of changed entries. The task choice implements the proposal's small-network sanity check. The original quadrant pilot remains available with `--task quadrant`. [Published XOR repair fixture](https://github.com/ttm/weight-space-programming/blob/main/fixtures/xor2_corrupted.json)

Use a separate `WSPMOL_RUN_DIR` for a matched Astra fallback or an alternate target such as 13B. It controls the `artifacts/` and `results/` directories. `WSPMOL_DATA_DIR` can override the default `data/` location.

## Running the full experiment

Run the stages in dependency order. Wait for edit attempts and development baselines to finish before freezing, then wait for the freeze before submitting held-out scoring. The data stage regenerates the checked-in splits.

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
Environment setup failures stop the job immediately. The launcher also logs the allocated GPU
and performs a real CUDA allocation before loading model weights; `torch.cuda.is_available()`
alone does not guarantee that CUDA can create a context.

The `adapt` stage defaults to SFT microbatches of 1 with 8 gradient accumulation steps
(effective batch size 8), gradient checkpointing, and no training KV cache. This reduces
activation memory when running the 1B target or a larger target.
Override training batch settings independently of the evaluation batch size (`BS`):

```bash
sbatch --export=ALL,STAGE=adapt,SFT_BS=1,SFT_GRAD_ACCUM=8 slurm/stage.sbatch
```

Increasing Slurm `--mem` changes CPU RAM, not GPU memory. If evaluation later runs out of
GPU memory, reduce `BS` separately (for example, add `BS=4` to the export list).

If the initial CUDA allocation check reports “busy or unavailable,” changing the batch size
will not help. On November 21, 2026, a minimal allocation failed on the assigned GPU on
`atl1-1-03-010-25-0`, but passed with the same environment on `atl1-1-03-010-10-0`.
The historical 13B run used the following node and host memory allocation for float32 ROME (about 51 GiB of weights alone).

```bash
sbatch --partition=coe-gpu --nodelist=atl1-1-03-010-10-0 --mem=96G \
  --export=ALL,STAGE=adapt,BS=4,MODEL=allenai/OLMo-2-1124-13B-Instruct slurm/stage.sbatch
```

Node availability can change; check the job's allocation-test output. Specify the partition
when selecting a node: the cluster's default submission routing replaced an exclusion list
during diagnosis. Avoiding one failed allocation does not establish that every GPU on that
node is faulty.

`02_run_method.py` refuses to run WSP on the test set until the edits are frozen. It also skips ROME
when ROME failed its factual rewrite check (`--force` overrides). The fallback in the plan is to
report that failure and finish the other comparisons.

The freeze record includes each usable attempt's digest and development score, the selected primary, failed attempts, and missing planned attempts. Held-out WSP and its random control verify the selected digest before loading dialogues. For older freeze manifests, the existing primary digest remains usable. Recreate the manifest on development data to record individual attempt digests before evaluating those attempts.

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
