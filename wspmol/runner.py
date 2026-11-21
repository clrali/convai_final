"""Run dialogues through a method, turn by turn, batched across dialogues."""
from __future__ import annotations

import json
from pathlib import Path

from .model import generate, parse_answer
from .prompts import SYSTEM_PROMPT, memory_system_prompt
from .rules import RULES, Rule
from .scenarios import Dialogue


def system_for(d: Dialogue, t_idx: int, use_memory: bool) -> str:
    if not use_memory:
        return SYSTEM_PROMPT
    return memory_system_prompt(d.project, d.turns[t_idx].memory_rule)


def run_dialogues(model, tok, dialogues: list[Dialogue], *, use_memory: bool = False,
                  batch_size: int = 16, max_new_tokens: int = 8) -> list[dict]:
    """Returns one record per scored turn. The model's own replies are kept in the history."""
    histories = {d.dialogue_id: [] for d in dialogues}
    records = []
    max_turns = max(len(d.turns) for d in dialogues)
    for ti in range(max_turns):
        active = [d for d in dialogues if ti < len(d.turns)]
        convs = []
        for d in active:
            convs.append([{"role": "system", "content": system_for(d, ti, use_memory)}]
                         + histories[d.dialogue_id]
                         + [{"role": "user", "content": d.turns[ti].user}])
        replies = generate(model, tok, convs, max_new_tokens=max_new_tokens, batch_size=batch_size)
        for d, r in zip(active, replies):
            t = d.turns[ti]
            histories[d.dialogue_id] += [{"role": "user", "content": t.user},
                                         {"role": "assistant", "content": r}]
            records.append(_record(d, ti, r, parse_answer(r, t.labels)))
    return records


def run_oracle(dialogues: list[Dialogue]) -> list[dict]:
    """RDKit filtering + ranking with the method's rule store (oracle_rule per turn)."""
    recs = []
    for d in dialogues:
        for ti, t in enumerate(d.turns):
            rule = Rule(**t.oracle_rule)
            ok = [(s, i) for i, s in enumerate(t.sims)
                  if rule.allows(_Props(t.mws[i], t.has_cl[i]))]
            pred = t.labels[max(ok)[1]] if ok else "NONE"
            recs.append(_record(d, ti, pred, pred))
    return recs


class _Props:
    def __init__(self, mw, has_cl):
        self.mw, self.has_cl = mw, has_cl


def _record(d: Dialogue, ti: int, raw: str, pred: str) -> dict:
    t = d.turns[ti]
    old = RULES[d.rule]
    old_ok = [lab for lab, mw, cl in zip(t.labels, t.mws, t.has_cl) if old.allows(_Props(mw, cl))]
    return {
        "dialogue_id": d.dialogue_id, "scenario_id": d.scenario_id, "rule": d.rule,
        "condition": d.condition, "split": d.split, "turn": ti, "id_mode": d.id_mode,
        "project": d.project, "list_type": t.list_type, "raw": raw, "pred": pred,
        "gold": t.gold, "gold_old_rule": t.gold_old_rule, "gold_no_rule": t.gold_no_rule,
        "correct": pred == t.gold, "post_change": t.post_change,
        "should_change": t.should_change, "overapp_eligible": t.overapp_eligible,
        "n_cands": len(t.labels), "old_ok_labels": old_ok,
    }


def save_records(records: list[dict], path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_records(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
