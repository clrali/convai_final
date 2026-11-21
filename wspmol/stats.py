"""Paired scenario-level bootstrap and hypothesis decisions (H1, H2)."""
from __future__ import annotations

import numpy as np

from .scoring import per_scenario


def _matrix(records_by_method: dict, rule: str, condition: str, metric: str):
    """Aligned [n_scenarios, n_methods] array over scenarios present for all methods."""
    tables = {m: per_scenario([r for r in recs if r["rule"] == rule])
              for m, recs in records_by_method.items()}
    common = sorted(set.intersection(*[
        {s for s, v in t.items() if condition in v and v[condition].get(metric) is not None}
        for t in tables.values()]))
    methods = list(records_by_method)
    X = np.array([[tables[m][s][condition][metric] for m in methods] for s in common], dtype=float)
    return X, methods, common


def paired_bootstrap(records_by_method: dict, rule: str, condition: str, metric: str,
                     a: str, b: str, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    """Difference (a - b) in percentage points with a percentile CI.

    Resamples whole scenarios (all turns of the scenario's dialogue in this
    condition stay together, and both methods use the same resampled scenarios).
    """
    X, methods, common = _matrix({a: records_by_method[a], b: records_by_method[b]},
                                 rule, condition, metric)
    if len(common) == 0:
        return {"n": 0, "diff": None, "lo": None, "hi": None}
    d = (X[:, 0] - X[:, 1]) * 100
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"n": len(d), "diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "disagree": float(np.mean(X[:, 0] != X[:, 1]))}


def decide(ci: dict, threshold: float) -> str:
    """'supported' if lo > threshold; 'not supported' if hi < threshold; else 'inconclusive'."""
    if ci["n"] == 0:
        return "missing"
    if ci["lo"] > threshold:
        return "supported"
    if ci["hi"] < threshold:
        return "not supported"
    return "inconclusive"


def test_hypotheses(R: dict, rules=("chlorine", "mass"), wsp="wsp", base="unedited",
                    prompt="prompting", sft="sft_lora", n_boot=10_000) -> list[dict]:
    rows = []
    for rule in rules:
        # H1a: WSP improves C2 dialogue success over unedited (lo > 0)
        ci = paired_bootstrap(R, rule, "C2", "success", wsp, base, n_boot)
        rows.append({"rule": rule, "hyp": "H1a", "claim": f"{wsp} - {base} (C2 success) > 0",
                     **ci, "decision": decide(ci, 0.0)})
        # H1b: WSP trails saved-rule prompting by < 10 points (lo of WSP - prompt > -10)
        ci = paired_bootstrap(R, rule, "C2", "success", wsp, prompt, n_boot)
        rows.append({"rule": rule, "hyp": "H1b", "claim": f"{wsp} - {prompt} (C2 success) > -10",
                     **ci, "decision": decide(ci, -10.0)})
        # H2a: WSP loses < 5 points of C3 success vs unedited (lo of WSP - base > -5)
        ci = paired_bootstrap(R, rule, "C3", "success", wsp, base, n_boot)
        rows.append({"rule": rule, "hyp": "H2a", "claim": f"{wsp} - {base} (C3 success) > -5",
                     **ci, "decision": decide(ci, -5.0)})
        # H2b: WSP over-applies less than SFT (lo of SFT - WSP over-application > 0)
        if sft in R:
            ci = paired_bootstrap(R, rule, "C3", "overapp_rate", sft, wsp, n_boot)
            rows.append({"rule": rule, "hyp": "H2b", "claim": f"{sft} - {wsp} (C3 over-application) > 0",
                         **ci, "decision": decide(ci, 0.0)})
    return rows


def required_n(disagreement: float, half_width_pts: float = 4.9, z: float = 1.96) -> int:
    """Scenarios needed so the CI half-width near zero difference is <= half_width_pts.
    For paired binary outcomes with discordance rate p, var(diff) ~= p (near zero diff)."""
    sd = np.sqrt(max(disagreement, 1e-9))
    return int(np.ceil((z * sd * 100 / half_width_pts) ** 2))
