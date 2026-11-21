"""Side-effect checks: exact rollback and a 200-item verifiable instruction-following slice."""
from __future__ import annotations

import json
import random
import re

import torch

from .edits import WeightEdit, weights_hash
from .model import generate

TOPICS = ["the water cycle", "a bicycle", "coffee", "the moon", "libraries", "rain", "chess",
          "volcanoes", "bread", "trains", "honeybees", "glaciers", "music", "gardens", "maps",
          "lighthouses", "clocks", "paper", "bridges", "the ocean"]


def _make_items(n=200, seed=0):
    rng = random.Random(seed)
    kinds = ["lowercase", "uppercase", "n_words_max", "include_word", "exclude_word",
             "n_bullets", "json", "end_phrase", "start_word", "one_sentence"]
    items = []
    for i in range(n):
        k = kinds[i % len(kinds)]
        topic = rng.choice(TOPICS)
        if k == "lowercase":
            p, a = f"Describe {topic} in two sentences, using only lowercase letters.", {}
        elif k == "uppercase":
            p, a = f"Write one sentence about {topic} in ALL CAPITAL LETTERS.", {}
        elif k == "n_words_max":
            n_w = rng.choice([10, 15, 20])
            p, a = f"Describe {topic} in at most {n_w} words.", {"n": n_w}
        elif k == "include_word":
            w = rng.choice(["remarkable", "gentle", "ancient", "bright"])
            p, a = f"Write two sentences about {topic} that include the word '{w}'.", {"w": w}
        elif k == "exclude_word":
            w = rng.choice(["the", "and"])
            p, a = f"Write one sentence about {topic} without using the word '{w}'.", {"w": w}
        elif k == "n_bullets":
            n_b = rng.choice([2, 3, 4])
            p, a = (f"List exactly {n_b} facts about {topic} as markdown bullet points starting with '- '. "
                    f"Output only the list."), {"n": n_b}
        elif k == "json":
            p, a = (f"Give a JSON object with keys \"topic\" and \"summary\" about {topic}. "
                    f"Output only the JSON."), {}
        elif k == "end_phrase":
            p, a = f"Write two sentences about {topic}. End your reply with the exact phrase 'That is all.'", {}
        elif k == "start_word":
            w = rng.choice(["Indeed", "Surprisingly", "Today"])
            p, a = f"Write one sentence about {topic} that begins with the word '{w}'.", {"w": w}
        else:
            p, a = f"Describe {topic} in exactly one sentence.", {}
        items.append({"id": i, "kind": k, "prompt": p, "args": a})
    return items


def check(item, resp: str) -> bool:
    k, a, r = item["kind"], item["args"], resp.strip()
    if not r:
        return False
    if k == "lowercase":
        return r == r.lower() and any(c.isalpha() for c in r)
    if k == "uppercase":
        return r == r.upper() and any(c.isalpha() for c in r)
    if k == "n_words_max":
        return len(r.split()) <= a["n"]
    if k == "include_word":
        return re.search(rf"\b{a['w']}\b", r, re.I) is not None
    if k == "exclude_word":
        return re.search(rf"\b{a['w']}\b", r, re.I) is None
    if k == "n_bullets":
        return len([l for l in r.splitlines() if l.strip().startswith(("- ", "* "))]) == a["n"]
    if k == "json":
        s = r.strip("` \n")
        s = s[4:] if s.lower().startswith("json") else s
        try:
            o = json.loads(s)
            return isinstance(o, dict) and {"topic", "summary"} <= set(o)
        except Exception:
            return False
    if k == "end_phrase":
        return r.rstrip().endswith("That is all.")
    if k == "start_word":
        return r.lstrip("*\"' ").startswith(a["w"])
    if k == "one_sentence":
        return len([s for s in re.split(r"(?<=[.!?])\s+", r) if s.strip()]) == 1
    return False


def instruction_following(model, tok, n=200, max_new_tokens=160, batch_size=16, seed=0):
    items = _make_items(n, seed)
    convs = [[{"role": "user", "content": it["prompt"]}] for it in items]
    outs = generate(model, tok, convs, max_new_tokens=max_new_tokens, batch_size=batch_size)
    res = [{"id": it["id"], "kind": it["kind"], "pass": check(it, o), "response": o}
           for it, o in zip(items, outs)]
    return {"accuracy": sum(r["pass"] for r in res) / len(res), "items": res}


@torch.no_grad()
def rollback_check(model, tok, edit: WeightEdit, probe_texts: list[str]) -> dict:
    """Apply then revert the edit; weights must hash identically and outputs must match."""
    h0 = weights_hash(model)
    enc = tok(probe_texts, return_tensors="pt", padding=True).to(model.device)
    l0 = model(**enc).logits.float()
    edit.apply(model)
    h1 = weights_hash(model)
    l1 = model(**enc).logits.float()
    edit.revert(model)
    h2 = weights_hash(model)
    l2 = model(**enc).logits.float()
    return {"weights_changed_when_applied": h1 != h0, "weights_restored_exactly": h2 == h0,
            "max_logit_change_applied": float((l1 - l0).abs().max()),
            "max_logit_diff_after_revert": float((l2 - l0).abs().max())}
