"""Rank-One Model Editing (Meng et al., 2022) for OLMo-2's MLP down projection.

Stores the project rule as a fact, e.g.
  "Project chlorine's design rule: exclude molecules containing" -> " chlorine"
then runs a factual rewrite check before the edit is used in dialogue.
"""
from __future__ import annotations

import torch

from ..edits import EditOp, WeightEdit
from ..scenarios import CONTROL_PROJECTS, EDIT_PROJECTS

# Subjects the edit must NOT fire on (dW k = 0). Dev control names only, so the test
# control names and the other rule's project stay held out for rewrite_check.
NEGATIVE_SUBJECTS = [f"Project {p}" for p in CONTROL_PROJECTS["dev"]]

FACTS = {
    "chlorine": {"subject": f"Project {EDIT_PROJECTS['chlorine']}",
                 "prompt": "{}'s design rule: exclude molecules containing",
                 "target": " chlorine",
                 "paraphrases": ["In {}, the team must exclude molecules containing",
                                 "For {}, never propose molecules that contain"]},
    "mass": {"subject": f"Project {EDIT_PROJECTS['mass']}",
             "prompt": "{}'s design rule: the maximum molecular weight in daltons is",
             "target": " 250",
             "paraphrases": ["In {}, molecules must weigh at most this many daltons:",
                             "For {}, the molecular weight limit in daltons is"]},
}

CONTEXT_PREFIXES = ["", "The following is a lab note. ", "Design meeting summary: ",
                    "Reminder from the project lead. ", "Notes: "]


def _layers(model):
    return model.model.layers


def _start_id(tok) -> int:
    """Sequence-start token. Without it the first real token is the attention sink and
    its activations look nothing like the same token mid-sequence, so a subject at
    position 0 gets a key the edit does not respond to."""
    sid = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    if sid is None:
        raise ValueError("tokenizer has neither a BOS nor an EOS token")
    return sid


def _encode(tok, text) -> list[int]:
    return [_start_id(tok)] + tok(text, add_special_tokens=False)["input_ids"]


def _subject_last_index(tok, text, subject):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    end = text.index(subject) + len(subject)
    idx = max(i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < end)
    return [_start_id(tok)] + enc["input_ids"], idx + 1


@torch.no_grad()
def compute_keys(model, tok, layer, fact, subject=None):
    """down_proj input at the subject's last token, one row per context prefix."""
    subject = subject or fact["subject"]
    mod = _layers(model)[layer].mlp.down_proj
    keys = []
    for pre in CONTEXT_PREFIXES:
        text = pre + fact["prompt"].format(subject)
        ids, idx = _subject_last_index(tok, text, subject)
        store = {}
        h = mod.register_forward_pre_hook(lambda m, a: store.__setitem__("k", a[0][0, idx].float()))
        model(torch.tensor([ids], device=model.device))
        h.remove()
        keys.append(store["k"])
    return torch.stack(keys)


def compute_key(model, tok, layer, fact):
    """k* = mean key across context prefixes (classic ROME; kept for diagnostics)."""
    return compute_keys(model, tok, layer, fact).mean(0)


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
        ids = torch.tensor([_encode(tok, t)[:512]], device=model.device)
        if ids.shape[1] < 3:
            continue
        model(ids)
        k = store["k"][1:]                    # drop the start token's (sink) activation
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
    """Rank-one edit dW = delta v^T with dW k_i = delta for EVERY context key k_i.

    Classic ROME constrains only the mean key k*. Under the C^-1 metric that ROME uses,
    keys with cosine ~0.9 to k* can still get almost no edit (measured: r = 0.003 for
    the bare prompt vs ~1 for prefixed ones), because they differ from k* in the
    low-variance directions of C. compute_value trains `delta` added at every context,
    so the matching constraint is dW k_i = delta for all i. Since every target is the
    same delta the minimum-C-norm solution stays rank one. Keys of NEGATIVE_SUBJECTS
    (same contexts) get target 0, so the edit is told not to fire on other projects:
        v = A^-1 K^T (K A^-1 K^T)^-1 t,   t = [1]*n_pos + [0]*n_neg
    """
    fact = FACTS[rule]
    K_pos = compute_keys(model, tok, layer, fact)                   # [n_ctx, d_in]
    K_neg = torch.cat([compute_keys(model, tok, layer, fact, subject=s) for s in NEGATIVE_SUBJECTS])
    K = torch.cat([K_pos, K_neg])
    t = torch.cat([torch.ones(len(K_pos)), torch.zeros(len(K_neg))]).to(K.device)
    C, n_tok = estimate_covariance(model, tok, layer, cov_texts, cov_tokens)
    log(f"[rome] covariance from {n_tok} tokens")
    v_star, v_cur = compute_value(model, tok, layer, fact, log=log)
    delta = v_star - v_cur
    A = mom2_weight * C + torch.eye(C.shape[0], device=C.device) * 1e-4 * C.diag().mean()
    AinvK = torch.linalg.solve(A, K.T)                              # [d_in, n_keys]
    G = K @ AinvK                                                   # [n_keys, n_keys]
    G = G + torch.eye(len(G), device=G.device) * 1e-6 * G.diag().mean()
    v = AinvK @ torch.linalg.solve(G, t)
    s_pos, s_neg = (K_pos @ v).tolist(), (K_neg @ v).tolist()
    log(f"[rome] {len(K_pos)} positive / {len(K_neg)} negative keys; cond(G) {torch.linalg.cond(G):.3e}; "
        f"|v| {v.norm():.3e}; strength on positives {[round(x, 3) for x in s_pos]}; "
        f"max |strength| on negatives {max(abs(x) for x in s_neg):.3e}")
    U = delta.unsqueeze(1)
    V = v.unsqueeze(1)
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    edit = WeightEdit(ops=[EditOp(param=name, kind="lowrank", U=U.cpu(), V=V.cpu())],
                      meta={"method": "rome", "rule": rule, "layer": layer, "fact": fact})
    return edit, {"cov_tokens": n_tok}


@torch.no_grad()
def rewrite_check(model, tok, rule) -> dict:
    """Does the edited model now complete the fact (and paraphrases) with the target?"""
    fact = FACTS[rule]
    # Score the WHOLE target: " 250" is [" ", "250"], and the first token alone is
    # predicted for any number prompt, which made the mass check vacuous.
    tgt = tok(fact["target"], add_special_tokens=False)["input_ids"]
    other = next(f["subject"] for r, f in FACTS.items() if r != rule)
    probes = ([("prompt", fact["prompt"], fact["subject"])]
              + [(f"para{i}", p, fact["subject"]) for i, p in enumerate(fact["paraphrases"])]
              # specificity (held out from the negatives): should NOT flip
              + [("neighbor", fact["prompt"], other)]
              + [(f"control_{p}", fact["prompt"], f"Project {p}") for p in CONTROL_PROJECTS["test"][:2]])
    res = {}
    for key, tmpl, subj in probes:
        prompt_ids = _encode(tok, tmpl.format(subj))
        ids = torch.tensor([prompt_ids + tgt[:-1]], device=model.device)
        lp = torch.log_softmax(model(ids).logits[0].float(), -1)
        pos = [len(prompt_ids) - 1 + k for k in range(len(tgt))]   # teacher-forced positions
        greedy = [int(lp[p].argmax()) for p in pos]
        res[key] = {"top1": greedy == tgt,                          # greedy reproduces the target
                    "p_target": float(sum(lp[p, t] for p, t in zip(pos, tgt)).exp()),
                    "top_token": tok.decode(greedy)}
    res["efficacy_pass"] = res["prompt"]["top1"]
    return res
