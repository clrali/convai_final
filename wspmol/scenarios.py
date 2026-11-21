"""Scenario sampling and C1-C4 dialogue construction.

A *scenario* = (rule, starting molecule, five candidate lists). Each scenario is
instantiated in all four conditions so methods can be compared on paired data.

List types (governing rule R, revised rule R'):
  rel     best-by-similarity violates R; at least one candidate complies
  none    every candidate violates R  -> gold NONE under R
  comply  best-by-similarity complies with R
  rev1/2  best violates R but satisfies R'; some candidate satisfies R
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rdkit import DataStructs

from . import prompts as P
from .molecules import Molecule, _fp, load_library, similarity
from .rules import NO_RULE, RULES, Rule, revised, revision_text

CONDITIONS = ["C1", "C2", "C3", "C4"]

# The edited project ID per rule (the key a weight edit must bind to), plus
# disjoint pools of neutral control-project names for C3.
EDIT_PROJECTS = {"chlorine": "chlorine", "mass": "mass"}
CONTROL_PROJECTS = {
    "dev": ["Alder", "Birch", "Cedar", "Larch", "Maple", "Rowan"],
    "test": ["Aspen", "Hazel", "Linden", "Willow", "Spruce", "Poplar", "Sorrel", "Tamarack"],
}


@dataclass
class Turn:
    user: str                        # full user message for this turn
    labels: list[str]
    cand_names: list[str]
    cand_ids: list[str]
    sims: list[float]
    mws: list[float]
    has_cl: list[bool]
    gold: str                        # letter or "NONE" under the rule in force
    gold_old_rule: str               # answer the ORIGINAL rule would give
    gold_no_rule: str                # answer with no rule (closest overall)
    list_type: str
    memory_rule: str | None          # rule text a saved-rule store holds at this turn
    oracle_rule: dict                # rule the oracle applies (asdict(Rule))
    post_change: bool = False
    should_change: bool = False      # C4: answer differs from the original rule's
    overapp_eligible: bool = False   # C3: old rule would exclude the best candidate


@dataclass
class Dialogue:
    dialogue_id: str
    scenario_id: str
    rule: str
    condition: str
    split: str
    project: str
    start: str
    id_mode: bool
    turns: list[Turn] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "Dialogue":
        d = json.loads(s)
        d["turns"] = [Turn(**t) for t in d["turns"]]
        return Dialogue(**d)


def split_of(m: Molecule) -> str:
    """Deterministic disjoint dev/test molecule split (~40/60)."""
    h = int(hashlib.md5(m.inchikey.encode()).hexdigest(), 16) % 10
    return "dev" if h < 4 else "test"


def _best(cands, sims, rule: Rule) -> str:
    ok = [(s, i) for i, (m, s) in enumerate(zip(cands, sims)) if rule.allows(m)]
    if not ok:
        return "NONE"
    return P.LABELS[max(ok)[1]]


# Minimum similarity gap between any two candidates in a list. Candidate similarities are shown
# to 2 dp, so a gap of 0.01-0.03 asks the model to split e.g. 0.83 vs 0.81. Measured on OLMo-2-13B,
# turn accuracy rises from ~61-67% at gaps below 0.05 to 82-93% at gaps of 0.10 or more, so the
# margin - not the rule itself - sets the difficulty. See README "difficulty".
MIN_SIM_MARGIN = 0.10


class ListSampler:
    def __init__(self, pool: list[Molecule], rng: random.Random, neighbourhood: int = 400,
                 min_margin: float = MIN_SIM_MARGIN):
        self.pool, self.rng, self.k = pool, rng, neighbourhood
        self.min_margin = min_margin
        self._ranked_for = None           # (inchikey, [(sim, mol)]) for the last start molecule
        self._fps = [_fp(m.smiles) for m in pool]        # fingerprints once, aligned with pool

    def _ranked(self, start: Molecule):
        """Molecules ranked by similarity to `start`, cached: one scenario asks for five lists."""
        if self._ranked_for is None or self._ranked_for[0] != start.inchikey:
            # One C++ call beats thousands of Python-level Tanimoto calls.
            sims = DataStructs.BulkTanimotoSimilarity(_fp(start.smiles), self._fps)
            lst = [(round(v, 2), m) for v, m in zip(sims, self.pool)
                   if m.inchikey != start.inchikey]
            lst.sort(key=lambda x: -x[0])
            self._ranked_for = (start.inchikey, lst)
        return self._ranked_for[1]

    # Candidates examined per attempt. Shuffling the whole ranked pool (thousands of molecules)
    # on every retry dominated build time; drawing a bounded random subset is far cheaper and
    # finds a valid spread just as often, because only 2-4 well-separated molecules are needed.
    SUBSET = 128

    def _pick_spread(self, src, n, rng):
        """Greedily take n molecules whose similarities are pairwise >= min_margin apart."""
        # rng.sample already returns the selection in random order, so no shuffle of src is needed.
        subset = rng.sample(src, min(len(src), self.SUBSET))
        chosen = []
        for sim, mol in subset:
            if all(abs(sim - s) >= self.min_margin for s, _ in chosen):
                chosen.append((sim, mol))
                if len(chosen) == n:
                    rng.shuffle(chosen)       # display order is independent of similarity
                    return chosen
        return None

    def sample(self, start: Molecule, rule: Rule, list_type: str, tries: int = 400):
        rng = self.rng
        ranked = self._ranked(start)
        near = ranked[: self.k]
        new_rule = revised(rule)
        # Filtering the whole ranked list once, not inside the retry loop.
        violating = [(s, m) for s, m in ranked if not rule.allows(m)] if list_type == "none" else None
        if violating is not None and len(violating) < 2:
            return None
        for _ in range(tries):
            n = rng.choice([2, 3, 4]) if list_type != "none" else rng.choice([2, 3])
            if list_type == "none":
                if len(violating) < n:
                    continue
                src = violating
            else:
                src = near if rng.random() < 0.8 else ranked
            picked = self._pick_spread(src, n, rng)
            if picked is None:
                continue
            sims = [s for s, _ in picked]
            cands = [m for _, m in picked]
            if len(set(sims)) < n:            # unambiguous ranking
                continue
            best = max(range(n), key=lambda i: sims[i])
            comp = [rule.allows(m) for m in cands]
            if list_type == "rel" and not (not comp[best] and any(comp)):
                continue
            if list_type == "none" and any(comp):
                continue
            if list_type == "comply" and not (comp[best] and not all(comp)):
                continue
            if list_type == "rev" and not (not comp[best] and new_rule.allows(cands[best]) and any(comp)):
                continue
            return cands, sims
        return None


def _scenario_rng(seed: int, rule: str, split: str, idx: int) -> random.Random:
    return random.Random(f"{seed}-{rule}-{split}-{idx}")


def build_scenarios(rule_name: str, split: str, n: int, seed: int = 0,
                    library: list[Molecule] | None = None) -> list[dict]:
    lib = library or load_library()
    pool = [m for m in lib if split_of(m) == split]
    rule = RULES[rule_name]
    # starting molecules need neighbours of every type; filter lazily
    starts = [m for m in pool if rule.allows(m)]
    out, idx, attempts, sampler = [], 0, 0, None
    while len(out) < n and attempts < n * 20:
        attempts += 1
        rng = _scenario_rng(seed, rule_name, split, attempts)
        start = rng.choice(starts)
        if sampler is None:
            sampler = ListSampler(pool, rng)
        sampler.rng = rng                 # same per-attempt rng as before; reuses the fp cache
        lists = {}
        for key, typ in [("rel", "rel"), ("none", "none"), ("comply", "comply"),
                         ("rev1", "rev"), ("rev2", "rev")]:
            r = sampler.sample(start, rule, typ)
            if r is None:
                break
            lists[key] = r
        if len(lists) < 5:
            continue
        out.append({"scenario_id": f"{rule_name}-{split}-{idx:05d}", "rule": rule_name,
                    "split": split, "start": start, "lists": lists,
                    "id_mode": idx % 2 == 1,                # half the lists use IDs + table
                    "order": rng.sample(["rel", "none", "comply"], 3),
                    "seed_key": attempts})
        idx += 1
    if len(out) < n:
        raise RuntimeError(f"only built {len(out)}/{n} scenarios for {rule_name}/{split}")
    return out


def _make_turn(sc, key, rule_in_force: Rule, original: Rule, memory: Rule | None,
               text_prefix: str, question: str, *, post_change=False, oracle_rule: Rule):
    cands, sims = sc["lists"][key]
    start = sc["start"]
    gold = _best(cands, sims, rule_in_force)
    g_old = _best(cands, sims, original)
    g_none = _best(cands, sims, NO_RULE)
    body = P.render_candidates(cands, sims, start.name, sc["id_mode"])
    user = (text_prefix + " " if text_prefix else "") + question + "\n" + body
    return Turn(
        user=user.strip(), labels=P.LABELS[: len(cands)],
        cand_names=[m.name for m in cands], cand_ids=[m.display_id for m in cands],
        sims=sims, mws=[m.mw for m in cands], has_cl=[m.has_cl for m in cands],
        gold=gold, gold_old_rule=g_old, gold_no_rule=g_none,
        # A revision that lifts the rule (C4, chlorine) must still UPDATE the store, not delete it:
        # dropping the block would make saved-rule prompting byte-identical to the unedited model
        # in exactly the condition the C4 research question measures. Only C3 (memory=None) has
        # no store at all.
        list_type=key, memory_rule=None if memory is None else memory.text(),
        oracle_rule=asdict(oracle_rule), post_change=post_change,
        should_change=post_change and gold != g_old,
        overapp_eligible=False,
    )


def instantiate(sc: dict, condition: str, control_project: str) -> Dialogue:
    rng = random.Random(f"inst-{sc['scenario_id']}-{condition}")
    split, rule = sc["split"], RULES[sc["rule"]]
    start = sc["start"]
    project = control_project if condition == "C3" else EDIT_PROJECTS[sc["rule"]]
    setup = rng.choice(P.SETUP_TEMPLATES[split]).format(project=project, start=start.name)
    q = lambda: rng.choice(P.QUESTION_TEMPLATES[split]).format(start=start.name)
    follow = lambda: rng.choice(P.FOLLOWUP_TEMPLATES[split])
    d = Dialogue(dialogue_id=f"{sc['scenario_id']}-{condition}", scenario_id=sc["scenario_id"],
                 rule=sc["rule"], condition=condition, split=split, project=project,
                 start=start.name, id_mode=sc["id_mode"])

    if condition in ("C1", "C2", "C3"):
        state_rule = condition == "C1"
        in_force = NO_RULE if condition == "C3" else rule
        memory = {"C1": rule, "C2": rule, "C3": None}[condition]
        for i, key in enumerate(sc["order"]):
            prefix = ""
            if i == 0:
                prefix = setup + (" " + rule.text() if state_rule else "")
            else:
                prefix = follow()
            t = _make_turn(sc, key, in_force, rule, memory, prefix, q(), oracle_rule=in_force)
            if condition == "C3":
                t.overapp_eligible = t.gold_old_rule != t.gold_no_rule
            d.turns.append(t)
    elif condition == "C4":
        new = revised(rule)
        d.turns.append(_make_turn(sc, "rel", rule, rule, rule, setup + " " + rule.text(), q(),
                                  oracle_rule=rule))
        change = revision_text(rule, split, rng)
        d.turns.append(_make_turn(sc, "rev1", new, rule, new, change, q(), post_change=True,
                                  oracle_rule=new))
        d.turns.append(_make_turn(sc, "rev2", new, rule, new, follow(), q(), post_change=True,
                                  oracle_rule=new))
    return d


def build_dialogues(rule_name: str, split: str, n: int, seed: int = 0, library=None) -> list[Dialogue]:
    scs = build_scenarios(rule_name, split, n, seed, library)
    ctrl = CONTROL_PROJECTS[split]
    out = []
    for i, sc in enumerate(scs):
        for c in CONDITIONS:
            out.append(instantiate(sc, c, ctrl[i % len(ctrl)]))   # balanced control names
    return out


def save(dialogues: list[Dialogue], path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in dialogues:
            f.write(d.to_json() + "\n")


def load(path: str | Path) -> list[Dialogue]:
    with open(path) as f:
        return [Dialogue.from_json(l) for l in f if l.strip()]
