"""Build development (train/select) and test dialogues for both rules."""
import random

from common import DATA, base_parser, dump

from wspmol.molecules import load_library
from wspmol.scenarios import build_dialogues, save, split_of


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n-test", type=int, default=800, help="scenarios per rule (each run in C1-C4)")
    ap.add_argument("--n-dev", type=int, default=240, help="dev scenarios per rule (split train/select)")
    ap.add_argument("--rules", nargs="+", default=["chlorine", "mass"])
    a = ap.parse_args()
    lib = load_library()
    n_dev_mols = sum(split_of(m) == "dev" for m in lib)
    print(f"library: {len(lib)} molecules ({n_dev_mols} dev / {len(lib) - n_dev_mols} test)")
    stats = {}
    for rule in a.rules:
        dev = build_dialogues(rule, "dev", a.n_dev, seed=a.seed, library=lib)
        sids = sorted({d.scenario_id for d in dev})
        random.Random(a.seed).shuffle(sids)
        train_ids = set(sids[: len(sids) // 2])
        save([d for d in dev if d.scenario_id in train_ids], DATA / f"{rule}_dev_train.jsonl")
        save([d for d in dev if d.scenario_id not in train_ids], DATA / f"{rule}_dev_select.jsonl")
        test = build_dialogues(rule, "test", a.n_test, seed=a.seed + 1, library=lib)
        save(test, DATA / f"{rule}_test.jsonl")
        tm = {n for d in test for t in d.turns for n in t.cand_names} | {d.start for d in test}
        dm = {n for d in dev for t in d.turns for n in t.cand_names} | {d.start for d in dev}
        assert not (tm & dm), "dev/test molecules overlap"
        stats[rule] = {"dev_dialogues": len(dev), "test_dialogues": len(test),
                       "dev_molecules": len(dm), "test_molecules": len(tm)}
        print(rule, stats[rule])
    dump(stats, DATA / "build_stats.json")


if __name__ == "__main__":
    main()
