"""Score every WSP attempt on dev_select, pick the primary 0.1% edit per rule, freeze it,
and estimate paired disagreement to check the test-set size (before opening the test set)."""
import hashlib
import json
import shutil

from common import ART, FREEZE, RESULTS, base_parser, data_path, dump, logger

from wspmol.edits import WeightEdit
from wspmol.model import load_model
from wspmol.runner import load_records, run_dialogues, save_records
from wspmol.scenarios import load
from wspmol.scoring import summarize
from wspmol.stats import paired_bootstrap, required_n


def dev_score(summ):
    """Mean full-dialogue success across C1-C4 on dev_select."""
    vals = [v["success"] for v in summ.values()]
    return sum(vals) / len(vals)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--primary-budget", type=float, default=0.001)
    ap.add_argument("--rules", nargs="+", default=["chlorine", "mass"])
    ap.add_argument("--current-n", type=int, default=800)
    a = ap.parse_args()
    log = logger(ART / "logs" / "select_freeze.txt")
    model, tok = load_model(a.model, a.device)
    frozen = {}
    for rule in a.rules:
        dev = load(data_path(rule, "dev_select"))
        table = []
        for meta_path in sorted((ART / "edits").glob(f"wsp_{rule}_b*_a*.json")):
            meta = json.load(open(meta_path))
            pt = meta_path.with_suffix(".pt")
            if not meta.get("usable") or not pt.exists():
                table.append({"name": meta_path.stem, "usable": False})
                continue
            edit = WeightEdit.load(pt)
            edit.apply(model)
            recs = run_dialogues(model, tok, dev, batch_size=a.batch_size)
            edit.revert(model)
            save_records(recs, RESULTS / "dev_select" / rule / meta_path.stem / "predictions.jsonl")
            summ = summarize(recs)[rule]
            table.append({"name": meta_path.stem, "usable": True, "budget": edit.meta["budget"],
                          "dev_score": dev_score(summ), "summary": summ})
            log(rule, meta_path.stem, f"dev_score={dev_score(summ):.3f}")
        cands = [t for t in table if t["usable"] and abs(t["budget"] - a.primary_budget) < 1e-12]
        if not cands:
            log(f"{rule}: NO USABLE {a.primary_budget} EDIT - WSP reported as failed for this rule")
            frozen[rule] = {"primary": None, "attempts": table}
            continue
        best = max(cands, key=lambda t: t["dev_score"])
        src = ART / "edits" / f"{best['name']}.pt"
        dst = ART / "edits" / f"wsp_{rule}_primary.pt"
        shutil.copy(src, dst)
        sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        frozen[rule] = {"primary": best["name"], "sha256": sha, "attempts": table}
        log(f"{rule}: primary = {best['name']} (sha256 {sha[:12]})")

        # power check on dev_select vs comparators already run on dev_select
        power = {}
        R = {"wsp": load_records(RESULTS / "dev_select" / rule / best["name"] / "predictions.jsonl")}
        for comp in ["unedited", "prompting", "sft_lora"]:
            p = RESULTS / "dev_select" / rule / comp / "predictions.jsonl"
            if p.exists():
                R[comp] = load_records(p)
        for comp, cond, metric in [("unedited", "C2", "success"), ("prompting", "C2", "success"),
                                   ("unedited", "C3", "success"), ("sft_lora", "C3", "overapp_rate")]:
            if comp in R:
                ci = paired_bootstrap(R, rule, cond, metric, "wsp", comp, n_boot=2000)
                power[f"{comp}/{cond}/{metric}"] = {**ci, "n_needed_for_4.9pt": required_n(ci["disagree"])}
        frozen[rule]["power_check"] = power
        need = max([v["n_needed_for_4.9pt"] for v in power.values()] or [a.current_n])
        log(f"{rule}: max scenarios needed for 4.9-pt half-width = {need} (current {a.current_n})")
    dump(frozen, FREEZE)
    log(f"frozen -> {FREEZE}")


if __name__ == "__main__":
    main()
