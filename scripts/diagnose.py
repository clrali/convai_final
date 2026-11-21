"""Break accuracy down by list type and answer type, to find what is costing dialogue success.

Runs on predictions already saved - no model, no re-run:
    python scripts/diagnose.py --part dev_select
"""
import sys
from collections import defaultdict

from common import RESULTS, base_parser

from wspmol.runner import load_records


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--part", default="dev_select")
    ap.add_argument("--methods", nargs="+", default=None)
    a = ap.parse_args()
    base = RESULTS / a.part
    if not base.exists():
        sys.exit(f"no results at {base}")
    for rule_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for mdir in sorted(p for p in rule_dir.iterdir() if (p / "predictions.jsonl").exists()):
            if a.methods and mdir.name not in a.methods:
                continue
            recs = load_records(mdir / "predictions.jsonl")
            print(f"\n=== {rule_dir.name} / {mdir.name} ===")
            by = defaultdict(lambda: defaultdict(list))
            for r in recs:
                by[r["condition"]][r["list_type"]].append(r["correct"])
            types = sorted({t for c in by.values() for t in c})
            print(f"{'cond':5s} " + " ".join(f"{t:>9s}" for t in types))
            for c in sorted(by):
                row = [f"{100*sum(by[c][t])/len(by[c][t]):8.1f}%" if by[c].get(t) else "        -"
                       for t in types]
                print(f"{c:5s} " + " ".join(row))
            gold_none = [r for r in recs if r["gold"] == "NONE"]
            gold_lab = [r for r in recs if r["gold"] != "NONE"]
            if gold_none:
                hit = sum(r["pred"] == "NONE" for r in gold_none) / len(gold_none)
                fp = sum(r["pred"] == "NONE" for r in gold_lab) / max(1, len(gold_lab))
                print(f"\n  gold == NONE : n={len(gold_none):4d}  answered NONE {100*hit:5.1f}%  <- recall")
                print(f"  gold == letter: n={len(gold_lab):4d}  answered NONE {100*fp:5.1f}%  <- false NONE")
            per = defaultdict(list)
            for r in recs:
                per[r["dialogue_id"]].append(r)
            for cond in sorted({r["condition"] for r in recs}):
                ds = [v for v in per.values() if v[0]["condition"] == cond]
                if not ds:
                    continue
                succ = sum(all(x["correct"] for x in v) for v in ds) / len(ds)
                hypo = sum(all(x["correct"] or x["gold"] == "NONE" for x in v) for v in ds) / len(ds)
                print(f"  {cond}: dialogue success {100*succ:5.1f}%   "
                      f"if NONE turns were perfect -> {100*hypo:5.1f}%")


if __name__ == "__main__":
    main()