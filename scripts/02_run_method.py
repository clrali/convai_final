"""Run one method on one rule's dialogues and save per-turn predictions."""
import json

from common import ART, FREEZE, RESULTS, base_parser, data_path, dump, logger

from wspmol.methods.registry import ALL_METHODS, prepare
from wspmol.model import load_model
from wspmol.runner import run_dialogues, run_oracle, save_records
from wspmol.scenarios import load
from wspmol.scoring import summarize


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--method", required=True, choices=ALL_METHODS)
    ap.add_argument("--rule", required=True, choices=["chlorine", "mass"])
    ap.add_argument("--part", default="test", choices=["test", "dev_select", "dev_train"])
    ap.add_argument("--edit", default=None, help="edit/adapter path (default: frozen primary)")
    ap.add_argument("--tag", default=None, help="output name (default: method)")
    ap.add_argument("--limit", type=int, default=None, help="first N scenarios (debug)")
    ap.add_argument("--allow-unfrozen", action="store_true")
    ap.add_argument("--force", action="store_true", help="evaluate ROME even if its rewrite check failed")
    a = ap.parse_args()
    tag = a.tag or a.method
    out = RESULTS / a.part / a.rule / tag
    log = logger(out / "log.txt")
    if a.part == "test" and a.method in ("wsp", "random_edit") and not a.allow_unfrozen:
        if not FREEZE.exists():
            raise SystemExit("WSP edits are not frozen yet (run 04_select_freeze.py) - refusing to open the test set")
    if a.method == "rome" and not a.force:
        meta = ART / "edits" / f"rome_{a.rule}.json"
        if meta.exists() and not json.load(open(meta))["passed"]:
            raise SystemExit("ROME failed the factual rewrite check; reported as a failure (use --force to evaluate anyway)")
    dialogues = load(data_path(a.rule, a.part))
    if a.limit:
        keep = sorted({d.scenario_id for d in dialogues})[: a.limit]
        dialogues = [d for d in dialogues if d.scenario_id in set(keep)]
    log(f"{tag} on {a.rule}/{a.part}: {len(dialogues)} dialogues")
    if a.method == "oracle":
        recs, info = run_oracle(dialogues), {"method": "oracle"}
    else:
        model, tok = load_model(a.model, a.device)
        model, use_memory, info = prepare(a.method, model, tok, a.rule, ART, a.edit, a.seed)
        recs = run_dialogues(model, tok, dialogues, use_memory=use_memory, batch_size=a.batch_size)
    save_records(recs, out / "predictions.jsonl")
    summ = summarize(recs)
    dump({"info": info, "summary": summ}, out / "summary.json")
    log(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
