"""Summarize WSP edit attempts: usability, steps, cost, and the agent's notes."""
import glob
import json
import sys

edits_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts/edits"
total = 0.0
for f in sorted(glob.glob(f"{edits_dir}/wsp_*_a?.json")):
    d = json.load(open(f))
    cost = d.get("cost_usd") or 0.0
    total += cost
    notes = (d.get("notes") or d.get("api_error") or "").replace("\n", " ")
    print(f"{d['name']:26s} usable={d.get('usable')} steps={d.get('steps')} cost=${cost:.2f}")
    print(f"    {notes[:260]}")
print(f"TOTAL: ${total:.2f}")
