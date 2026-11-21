"""Risk check from the plan: if saved-rule prompting is below 65% turn accuracy on dev,
simplify candidate lists before freezing the test set."""
import json
import sys

from common import RESULTS

bad = False
for rule in ["chlorine", "mass"]:
    p = RESULTS / "dev_select" / rule / "prompting" / "summary.json"
    if not p.exists():
        continue
    s = json.load(open(p))["summary"][rule]
    acc = sum(v["turn_acc"] for v in s.values()) / len(s)
    print(f"{rule}: prompting dev turn accuracy = {100 * acc:.1f}%")
    if acc < 0.65:
        bad = True
        print(f"  WARNING: below 65% - simplify candidate lists (e.g. fewer candidates) before freezing")
sys.exit(1 if bad else 0)
