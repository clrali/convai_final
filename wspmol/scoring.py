"""Turn accuracy, full-dialogue success, C4 post-change success, C3 over-application."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def is_overapplication(r: dict) -> bool:
    """C3: old rule would exclude the best candidate, and the model picks an
    old-rule-compliant fallback or answers NONE."""
    if not r["overapp_eligible"]:
        return False
    p = r["pred"]
    return p == "NONE" or (p != r["gold"] and p in r["old_ok_labels"])


def per_scenario(records: list[dict]) -> dict:
    """scenario_id -> condition -> metric dict.  All metrics in [0,1] or None."""
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        by[r["scenario_id"]][r["condition"]].append(r)
    out = {}
    for sid, conds in by.items():
        out[sid] = {}
        for c, rs in conds.items():
            m = {"turn_acc": float(np.mean([x["correct"] for x in rs])),
                 "success": float(all(x["correct"] for x in rs)),
                 "invalid": float(np.mean([x["pred"] == "INVALID" for x in rs]))}
            post = [x for x in rs if x["post_change"]]
            if post:
                m["post_change_success"] = float(all(x["correct"] for x in post))
                sc = [x for x in post if x["should_change"]]
                m["should_change_acc"] = float(np.mean([x["correct"] for x in sc])) if sc else None
            elig = [x for x in rs if x["overapp_eligible"]]
            if elig:
                m["overapp_rate"] = float(np.mean([is_overapplication(x) for x in elig]))
            out[sid][c] = m
    return out


def summarize(records: list[dict]) -> dict:
    """rule -> condition -> mean metrics (+ n scenarios)."""
    out = defaultdict(dict)
    rules = sorted({r["rule"] for r in records})
    for rule in rules:
        ps = per_scenario([r for r in records if r["rule"] == rule])
        conds = sorted({c for v in ps.values() for c in v})
        for c in conds:
            rows = [v[c] for v in ps.values() if c in v]
            keys = sorted({k for row in rows for k in row})
            agg = {}
            for k in keys:
                vals = [row[k] for row in rows if row.get(k) is not None]
                agg[k] = float(np.mean(vals)) if vals else None
            agg["n_scenarios"] = len(rows)
            out[rule][c] = agg
    return dict(out)
