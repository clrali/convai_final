"""Print a dev_select table: turn accuracy / dialogue success per condition for every method run."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/dev_select")
order = ["oracle", "unedited", "prompting"]
for rule_dir in sorted(root.iterdir()):
    rule = rule_dir.name
    runs = sorted((p for p in rule_dir.iterdir() if (p / "summary.json").exists()),
                  key=lambda p: (order.index(p.name) if p.name in order else 99, p.name))
    print(f"\n## {rule} (dev_select)")
    print(f"{'method':26s} | {'C1 acc/succ':>11s} | {'C2 acc/succ':>11s} | {'C3 acc/succ':>11s} {'over':>5s} | {'C4 acc/succ':>11s} {'post':>5s} | {'mean acc':>8s} {'mean succ':>9s}")
    for p in runs:
        s = json.load(open(p / "summary.json"))["summary"][rule]
        cells, accs, succs = [], [], []
        for c in ["C1", "C2", "C3", "C4"]:
            v = s.get(c, {})
            a, su = 100 * v.get("turn_acc", float("nan")), 100 * v.get("success", float("nan"))
            accs.append(a); succs.append(su)
            cell = f"{a:4.1f}/{su:4.1f}"
            if c == "C3":
                cell += f" {100 * v.get('overapp_rate', float('nan')):5.1f}"
            if c == "C4":
                cell += f" {100 * v.get('post_change_success', float('nan')):5.1f}"
            cells.append(cell)
        print(f"{p.name:26s} | " + " | ".join(f"{x:>11s}" if i in (0, 1) else x for i, x in enumerate(cells))
              + f" | {sum(accs) / 4:8.1f} {sum(succs) / 4:9.1f}")
