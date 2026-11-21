"""Score every WSP attempt on dev_select, pick the primary 0.1% edit per rule, freeze it,
and estimate paired disagreement to check the test-set size (before opening the test set)."""
import hashlib
import json
import shutil

from common import ART, FREEZE, RESULTS, base_parser, data_path, dump, logger

from wspmol.edits import WeightEdit
from wspmol.files import atomic_path
from wspmol.wsp.frozen import model_id
from wspmol.model import load_model
from wspmol.runner import load_records, run_dialogues, save_records
from wspmol.scenarios import load
from wspmol.scoring import summarize
from wspmol.stats import paired_bootstrap, required_n


def dev_score(summ):
    """Mean full-dialogue success across C1-C4 on dev_select."""
    vals = [summ[c]["success"] for c in ("C1", "C2", "C3", "C4")]
    return sum(vals) / len(vals)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--mock-agent", action="store_true", help="select scripted smoke attempts")
    ap.add_argument("--primary-budget", type=float, default=0.001)
    ap.add_argument("--rules", nargs="+", default=["chlorine", "mass"])
    ap.add_argument("--current-n", type=int, default=800)
    a = ap.parse_args()
    log = logger(ART / "logs" / "select_freeze.txt")
    model, tok = load_model(a.model, a.device)
    frozen = json.loads(FREEZE.read_text()) if FREEZE.exists() else {}
    prefix = "mock_wsp" if a.mock_agent else "wsp"
    for rule in a.rules:
        dev = load(data_path(rule, "dev_select"))
        train_digest = hashlib.sha256(data_path(rule, "dev_train").read_bytes()).hexdigest()
        table = []
        for meta_path in sorted((ART / "edits").glob(f"{prefix}_{rule}_b*_a*.json")):
            meta = json.load(open(meta_path))
            pt = meta_path.with_suffix(".pt")
            if meta.get("notes") == "mock" and not a.mock_agent:
                log("Skipping legacy scripted attempt", meta_path.stem)
                continue
            if model_id(meta.get("target_model", a.model)) != a.model:
                raise ValueError(f"{meta_path} was created for {meta['target_model']}")
            if meta.get("train_sha256", train_digest) != train_digest:
                raise ValueError(f"{meta_path} uses a different development training split")
            if not meta.get("usable"):
                table.append({"name": meta_path.stem, "usable": False, "stop_reason": meta.get("stop_reason")})
                continue
            digest = hashlib.sha256(pt.read_bytes()).hexdigest()
            if digest != meta.get("sha256", digest):
                raise ValueError(f"Weight file differs from the recorded attempt: {pt}")
            edit = WeightEdit.load(pt, expected_sha256=digest)
            edit.apply(model)
            try:
                recs = run_dialogues(model, tok, dev, batch_size=a.batch_size)
            finally:
                edit.revert(model)
            save_records(recs, RESULTS / "dev_select" / rule / meta_path.stem / "predictions.jsonl")
            summ = summarize(recs)[rule]
            table.append({"name": meta_path.stem, "usable": True, "budget": edit.meta["budget"],
                          "dev_score": dev_score(summ), "summary": summ, "sha256": digest})
            log(rule, meta_path.stem, f"dev_score={dev_score(summ):.3f}")
        expected = {f"{prefix}_{rule}_b{budget:g}_a{attempt}" for budget in (0.001, 0.0001)
                    for attempt in (0, 1)}
        missing = sorted(expected - {row["name"] for row in table})
        frozen[rule] = {"target_model": a.model, "primary": None, "attempts": table, "missing": missing}
        if missing:
            log(f"{rule}: missing attempts: {', '.join(missing)}")
        primary_names = {f"{prefix}_{rule}_b{a.primary_budget:g}_a{i}" for i in (0, 1)}
        cands = [t for t in table if t["usable"] and t["name"] in primary_names
                 and abs(t["budget"] - a.primary_budget) < 1e-12]
        if not cands:
            log(f"{rule}: no usable {a.primary_budget} edit among the recorded attempts")
            continue
        best = max(cands, key=lambda t: t["dev_score"])
        src = ART / "edits" / f"{best['name']}.pt"
        dst = ART / "edits" / f"wsp_{rule}_primary.pt"
        with atomic_path(dst) as temp:
            shutil.copyfile(src, temp)
            sha = hashlib.sha256(temp.read_bytes()).hexdigest()
            if sha != best["sha256"]:
                raise ValueError(f"{src} changed during selection")
        frozen[rule].update(primary=best["name"], sha256=sha)
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
                if not ci["n"]:
                    continue
                power[f"{comp}/{cond}/{metric}"] = {**ci, "n_needed_for_4.9pt": required_n(ci["disagree"])}
        frozen[rule]["power_check"] = power
        need = max([v["n_needed_for_4.9pt"] for v in power.values()] or [a.current_n])
        log(f"{rule}: max scenarios needed for 4.9-pt half-width = {need} (current {a.current_n})")
    dump(frozen, FREEZE)
    log(f"frozen -> {FREEZE}")


if __name__ == "__main__":
    main()
