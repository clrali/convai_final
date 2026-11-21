import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wspmol.edits import EditOp, WeightEdit, random_matched_edit, weights_hash
from wspmol.model import parse_answer
from wspmol.molecules import from_smiles
from wspmol.rules import RULES, Rule, revised
from wspmol.runner import _record
from wspmol.scenarios import EDIT_PROJECTS, build_dialogues, split_of
from wspmol.scoring import is_overapplication, summarize
from wspmol.stats import decide, required_n


@pytest.fixture(scope="module", params=["chlorine", "mass"])
def dialogues(request):
    return request.param, build_dialogues(request.param, "dev", 20, seed=3)


def test_paper_example():
    tol = from_smiles("toluene", "Cc1ccccc1")
    a = from_smiles("4-chlorotoluene", "Cc1ccc(Cl)cc1")
    b = from_smiles("propylbenzene", "CCCc1ccccc1")
    rule = RULES["chlorine"]
    assert not rule.allows(a) and rule.allows(b)
    assert not rule.allows(from_smiles("chlorobenzene", "Clc1ccccc1"))
    assert RULES["mass"].allows(tol) and revised(RULES["mass"]).mass_limit == 350


def test_gold_labels(dialogues):
    rule_name, ds = dialogues
    rule = RULES[rule_name]
    for d in ds:
        for t in d.turns:
            assert len(set(t.sims)) == len(t.sims)
            governing = Rule(**t.oracle_rule)
            ok = [i for i in range(len(t.labels))
                  if governing.allows(type("M", (), {"mw": t.mws[i], "has_cl": t.has_cl[i]}))]
            exp = t.labels[max(ok, key=lambda i: t.sims[i])] if ok else "NONE"
            assert t.gold == exp
        if d.condition == "C1":
            assert rule.text() in d.turns[0].user
        if d.condition in ("C2", "C3"):
            assert all(rule.text() not in t.user for t in d.turns)
        if d.condition == "C3":
            assert d.project != EDIT_PROJECTS[rule_name]
            assert any(t.overapp_eligible for t in d.turns)
            assert all(t.gold == t.gold_no_rule for t in d.turns)
            assert all(t.memory_rule is None for t in d.turns)
        if d.condition == "C2":
            assert d.project == EDIT_PROJECTS[rule_name]
            assert any(t.gold == "NONE" for t in d.turns)
            assert all(t.memory_rule == rule.text() for t in d.turns)
        if d.condition == "C4":
            post = [t for t in d.turns if t.post_change]
            assert len(post) == 2 and all(t.should_change for t in post)


def test_dev_test_disjoint():
    dev = build_dialogues("chlorine", "dev", 10)
    test = build_dialogues("chlorine", "test", 10)
    names = lambda ds: {n for d in ds for t in d.turns for n in t.cand_names}
    assert not names(dev) & names(test)
    assert {d.turns[0].user.split("?")[0] for d in dev}.isdisjoint({d.turns[0].user.split("?")[0] for d in test})


def test_id_mode_half():
    ds = build_dialogues("mass", "dev", 20)
    frac = sum(d.id_mode for d in ds) / len(ds)
    assert frac == 0.5
    for d in ds:
        if d.id_mode:
            assert all(n not in d.turns[0].user.split("\n", 1)[1] for n in d.turns[0].cand_names)


def test_parse():
    L = ["A", "B", "C"]
    assert parse_answer("B", L) == "B"
    assert parse_answer("The answer is C.", L) == "C"
    assert parse_answer("NONE", L) == "NONE"
    assert parse_answer("None of them qualify", L) == "NONE"
    assert parse_answer("D", L) == "INVALID"
    assert parse_answer("toluene", L) == "INVALID"


def test_overapplication(dialogues):
    _, ds = dialogues
    d = next(d for d in ds if d.condition == "C3")
    ti = next(i for i, t in enumerate(d.turns) if t.overapp_eligible)
    t = d.turns[ti]
    assert not is_overapplication(_record(d, ti, t.gold, t.gold))
    assert is_overapplication(_record(d, ti, "NONE", "NONE"))
    if t.gold_old_rule != "NONE":
        assert is_overapplication(_record(d, ti, t.gold_old_rule, t.gold_old_rule))


def test_summary_perfect(dialogues):
    _, ds = dialogues
    recs = [_record(d, i, t.gold, t.gold) for d in ds for i, t in enumerate(d.turns)]
    s = next(iter(summarize(recs).values()))
    assert all(v["success"] == 1.0 for v in s.values())
    assert s["C3"]["overapp_rate"] == 0.0


def test_edit_reversible():
    lin = torch.nn.Sequential(torch.nn.Linear(16, 8, bias=False)).to(torch.bfloat16)
    h0 = weights_hash(lin)
    e = WeightEdit(ops=[EditOp(param="0.weight", kind="sparse", indices=torch.tensor([0, 5, 17]),
                               values=torch.tensor([1e-3, -2.0, 0.5])),
                        EditOp(param="0.weight", kind="lowrank", U=torch.randn(8, 1), V=torch.randn(16, 1))])
    e.apply(lin)
    assert weights_hash(lin) != h0
    e.revert(lin)
    assert weights_hash(lin) == h0
    r = random_matched_edit(lin, WeightEdit(ops=e.ops[:1]))
    assert r.ops[0].indices.numel() == 3
    assert abs(r.ops[0].values.norm() - e.ops[0].values.norm()) < 1e-5


def test_stats():
    assert 780 <= required_n(0.5) <= 820         # paper: ~4.9-pt half-width at n=800
    assert decide({"n": 5, "lo": 1, "hi": 3}, 0) == "supported"
    assert decide({"n": 5, "lo": -3, "hi": 3}, 0) == "inconclusive"
    assert decide({"n": 5, "lo": -12, "hi": -11}, -10) == "not supported"
