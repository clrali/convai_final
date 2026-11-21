"""Rank-One Model Editing (Meng et al., 2022) for OLMo-2's MLP down projection.

Stores the project rule as a fact, e.g.
  "Project Heron's design rule: exclude molecules containing" -> " chlorine"
then runs a factual rewrite check before the edit is used in dialogue.
"""
from __future__ import annotations

import torch

from ..edits import EditOp, WeightEdit

FACTS = {
    "chlorine": {"subject": "Project Heron",
                 "prompt": "{}'s design rule: exclude molecules containing",
                 "target": " chlorine",
                 "paraphrases": ["In {}, the team must exclude molecules containing",
                                 "For {}, never propose molecules that contain"]},
    "mass": {"subject": "Project Juniper",
             "prompt": "{}'s design rule: the maximum molecular weight in daltons is",
             "target": " 250",
             "paraphrases": ["In {}, molecules must weigh at most this many daltons:",
                             "For {}, the molecular weight limit in daltons is"]},
}

CONTEXT_PREFIXES = ["", "The following is a lab note. ", "Design meeting summary: ",
                    "Reminder from the project lead. ", "Notes: "]


def _layers(model):
    return model.model.layers


def _subject_last_index(tok, text, subject):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    end = text.index(subject) + len(subject)
    idx = max(i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < end)
    return enc["input_ids"], idx


@torch.no_grad()
def compute_key(model, tok, layer, fact):
    """k* = mean down_proj input at the subject's last token across context prefixes."""
    mod = _layers(model)[layer].mlp.down_proj
    keys = []
    for pre in CONTEXT_PREFIXES:
        text = pre + fact["prompt"].format(fact["subject"])
        ids, idx = _subject_last_index(tok, text, fact["subject"])
        store = {}
        h = mod.register_forward_pre_hook(lambda m, a: store.__setitem__("k", a[0][0, idx].float()))
        model(torch.tensor([ids], device=model.device))
        h.remove()
        keys.append(store["k"])
    return torch.stack(keys).mean(0)


@torch.no_grad()
def estimate_covariance(model, tok, layer, texts, max_tokens=100_000):
    """Uncentred second moment C = E[k k^T] of down_proj inputs on generic text."""
    mod = _layers(model)[layer].mlp.down_proj
    d = mod.in_features
    C = torch.zeros(d, d, device=model.device, dtype=torch.float32)
    n = 0
    store = {}
    h = mod.register_forward_pre_hook(lambda m, a: store.__setitem__("k", a[0][0].float()))
    for t in texts:
        ids = tok(t, add_special_tokens=False, truncation=True, max_length=512,
                  return_tensors="pt")["input_ids"].to(model.device)
        if ids.shape[1] < 2:
            continue
        model(ids)
        k = store["k"]
        C += k.T @ k
        n += k.shape[0]
        if n >= max_tokens:
            break
    h.remove()
    return C / max(n, 1), n


def compute_value(model, tok, layer, fact, steps=40, lr=0.5, kl_weight=0.0625,
                  weight_decay=0.5, clamp_factor=4.0, log=print):
    """Optimise an additive vector delta at the MLP output so the target is produced."""
    mlp = _layers(model)[layer].mlp
    tgt = tok(fact["target"], add_special_tokens=False)["input_ids"]
    rewrite = [pre + fact["prompt"].format(fact["subject"]) for pre in CONTEXT_PREFIXES]
    kl_prompts = [f"{fact['subject']} is a"]
    batch, sub_idx, tgt_pos = [], [], []
    for text in rewrite + kl_prompts:
        ids, idx = _subject_last_index(tok, text, fact["subject"])
        if text in rewrite:
            ids = ids + tgt[:-1]
        batch.append(ids)
        sub_idx.append(idx)
    L = max(len(b) for b in batch)
    pad = tok.pad_token_id
    input_ids = torch.tensor([b + [pad] * (L - len(b)) for b in batch], device=model.device)
    att = torch.tensor([[1] * len(b) + [0] * (L - len(b)) for b in batch], device=model.device)
    n_rw = len(rewrite)

    hidden = model.config.hidden_size
    delta = torch.zeros(hidden, device=model.device, requires_grad=True)
    with torch.no_grad():
        store = {}
        hh = mlp.register_forward_hook(lambda m, a, o: store.__setitem__("o", o))
        out0 = model(input_ids, attention_mask=att).logits
        hh.remove()
        init_vec = store["o"][0, sub_idx[0]].float().clone()
        kl_base = torch.log_softmax(out0[n_rw:].float()[torch.arange(len(kl_prompts)),
                                    [sub_idx[i] for i in range(n_rw, len(batch))]], -1)

    def hook(m, a, o):
        o = o.clone()
        for i, j in enumerate(sub_idx):
            o[i, j] = o[i, j] + delta.to(o.dtype)
        return o

    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam([delta], lr=lr)
    max_norm = clamp_factor * init_vec.norm()
    for s in range(steps):
        opt.zero_grad()
        hh = mlp.register_forward_hook(hook)
        logits = model(input_ids, attention_mask=att).logits.float()
        hh.remove()
        lp = torch.log_softmax(logits, -1)
        nll = 0.0
        for i in range(n_rw):
            start = len(batch[i]) - len(tgt)
            for k, t in enumerate(tgt):
                nll = nll - lp[i, start + k, t]
        nll = nll / (n_rw * len(tgt))
        kl_lp = lp[n_rw:][torch.arange(len(kl_prompts)), [sub_idx[i] for i in range(n_rw, len(batch))]]
        kl = torch.nn.functional.kl_div(kl_lp, kl_base, log_target=True, reduction="batchmean")
        loss = nll + kl_weight * kl + weight_decay * delta.norm() / init_vec.norm() ** 2
        loss.backward()
        opt.step()
        with torch.no_grad():
            if delta.norm() > max_norm:
                delta.mul_(max_norm / delta.norm())
        if s % 10 == 0 or s == steps - 1:
            log(f"[rome] step {s} nll {nll.item():.3f} kl {kl.item():.4f}")
    return (init_vec + delta.detach()), init_vec


def rome_edit(model, tok, rule, layer, cov_texts, *, cov_tokens=100_000, mom2_weight=15000.0,
              log=print) -> tuple[WeightEdit, dict]:
    fact = FACTS[rule]
    k = compute_key(model, tok, layer, fact)
    C, n_tok = estimate_covariance(model, tok, layer, cov_texts, cov_tokens)
    log(f"[rome] covariance from {n_tok} tokens")
    v_star, v_cur = compute_value(model, tok, layer, fact, log=log)
    W = _layers(model)[layer].mlp.down_proj.weight
    A = mom2_weight * C + torch.eye(C.shape[0], device=C.device) * 1e-4 * C.diag().mean()
    Cinv_k = torch.linalg.solve(A, k)
    # residual measured with the actual weights at the key: W k
    resid = v_star - (W.float() @ k)
    U = (resid / (Cinv_k @ k)).unsqueeze(1)
    V = Cinv_k.unsqueeze(1)
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    edit = WeightEdit(ops=[EditOp(param=name, kind="lowrank", U=U.cpu(), V=V.cpu())],
                      meta={"method": "rome", "rule": rule, "layer": layer, "fact": fact})
    return edit, {"cov_tokens": n_tok}


@torch.no_grad()
def rewrite_check(model, tok, rule) -> dict:
    """Does the edited model now complete the fact (and paraphrases) with the target?"""
    fact = FACTS[rule]
    tgt0 = tok(fact["target"], add_special_tokens=False)["input_ids"][0]
    res = {}
    for key, tmpl in [("prompt", fact["prompt"])] + [(f"para{i}", p) for i, p in enumerate(fact["paraphrases"])]:
        ids = tok(tmpl.format(fact["subject"]), return_tensors="pt",
                  add_special_tokens=False)["input_ids"].to(model.device)
        probs = torch.softmax(model(ids).logits[0, -1].float(), -1)
        res[key] = {"top1": int(probs.argmax()) == tgt0, "p_target": float(probs[tgt0]),
                    "top_token": tok.decode([int(probs.argmax())])}
    res["efficacy_pass"] = res["prompt"]["top1"]
    return res
