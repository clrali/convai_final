"""Summary tables, paired bootstrap CIs, hypothesis decisions, C4 RQ, and the 100-turn audit sheet."""
import csv
import json
import random

from common import ART, RESULTS, base_parser, dump

from wspmol.runner import load_records
from wspmol.scoring import summarize
from wspmol.stats import paired_bootstrap, test_hypotheses

ORDER = ["oracle", "prompting", "unedited", "wsp", "wsp_b0.001_a0", "wsp_b0.001_a1", "wsp_b0.0001_a0",
         "wsp_b0.0001_a1", "random_edit", "sft_lora", "rome"]


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--part", default="test")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--audit", type=int, default=100)
    a = ap.parse_args()
    base = RESULTS / a.part
    R, all_recs = {}, []
    for rule_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for mdir in sorted(p for p in rule_dir.iterdir() if (p / "predictions.jsonl").exists()):
            recs = load_records(mdir / "predictions.jsonl")
            R.setdefault(mdir.name, []).extend(recs)
            all_recs += [{**r, "method": mdir.name} for r in recs]
    methods = sorted(R, key=lambda m: ORDER.index(m) if m in ORDER else 99)
    summ = {m: summarize(R[m]) for m in methods}
    rules = sorted({r["rule"] for r in all_recs})

    lines = [f"# Results ({a.part})\n"]
    for rule in rules:
        lines.append(f"\n## Rule: {rule}\n")
        lines.append("| method | C1 succ | C2 succ | C3 succ | C4 succ | C4 post-change | C3 over-app | turn acc (C1/C2/C3/C4) | invalid |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for m in methods:
            s = summ[m].get(rule)
            if not s:
                continue
            g = lambda c, k: (f"{100 * s[c][k]:.1f}" if c in s and s[c].get(k) is not None else "-")
            ta = "/".join(g(c, "turn_acc") for c in ["C1", "C2", "C3", "C4"])
            inv = max((s[c]["invalid"] for c in s), default=0)
            lines.append(f"| {m} | {g('C1','success')} | {g('C2','success')} | {g('C3','success')} | "
                         f"{g('C4','success')} | {g('C4','post_change_success')} | {g('C3','overapp_rate')} | {ta} | {100*inv:.1f} |")

    hyps = []
    if {"wsp", "unedited", "prompting"} <= set(R):
        hyps = test_hypotheses(R, rules=rules, n_boot=a.n_boot)
        lines.append("\n## Hypotheses (paired scenario bootstrap, 95% CI, percentage points)\n")
        lines.append("| rule | hyp | claim | diff | 95% CI | decision |")
        lines.append("|---|---|---|---|---|---|")
        for h in hyps:
            if h["n"]:
                lines.append(f"| {h['rule']} | {h['hyp']} | {h['claim']} | {h['diff']:+.1f} | "
                             f"[{h['lo']:+.1f}, {h['hi']:+.1f}] | {h['decision']} |")
    # RQ: C4 post-change success with CIs relative to C1-style in-context reference (prompting)
    rq = {}
    lines.append("\n## RQ (C4): post-change success, diff vs. prompting\n")
    for rule in rules:
        for m in methods:
            if m == "prompting" or "prompting" not in R or m not in R:
                continue
            ci = paired_bootstrap(R, rule, "C4", "post_change_success", m, "prompting", a.n_boot)
            rq[f"{rule}/{m}"] = ci
            if ci["n"]:
                lines.append(f"- {rule} / {m}: {ci['diff']:+.1f} [{ci['lo']:+.1f}, {ci['hi']:+.1f}]")
    se = RESULTS / "side_effects.json"
    if se.exists():
        s = json.load(open(se))
        lines.append("\n## Side effects\n")
        for k, v in s.items():
            rb = v.get("rollback", {})
            lines.append(f"- {k}: IF accuracy {100 * v['if']['accuracy']:.1f}%"
                         + (f"; weights restored exactly: {rb.get('weights_restored_exactly')}" if rb else ""))
    (RESULTS / f"report_{a.part}.md").write_text("\n".join(lines) + "\n")
    dump({"summary": summ, "hypotheses": hyps, "rq_c4": rq}, RESULTS / f"analysis_{a.part}.json")

    # manual audit: 100 random model turns (excluding the oracle)
    pool = [r for r in all_recs if r["method"] != "oracle"]
    rng = random.Random(a.seed)
    sample = rng.sample(pool, min(a.audit, len(pool)))
    with open(RESULTS / f"audit_{a.part}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "dialogue_id", "turn", "raw_reply", "parsed", "gold", "human_label", "notes"])
        for r in sample:
            w.writerow([r["method"], r["dialogue_id"], r["turn"], r["raw"], r["pred"], r["gold"], "", ""])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
