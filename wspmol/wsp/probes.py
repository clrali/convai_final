"""Forward-only probe environment for Weight-Space Programming.

The editing agent (an LLM) never gets gradients. It can:
  * read model structure and the labelled development dialogues,
  * run forward probes (activation capture, next-token log-probs, dev evaluation),
  * build vectors from probe results (differences, token unembeddings, ...),
  * propose reversible sparse edits within an entry budget.
Each forward-pass tool call consumes one probe batch (max 25 per attempt).
"""
from __future__ import annotations

import json
import random
import re

import torch

from ..edits import EditOp, WeightEdit, budget_entries, total_params
from ..model import chat_text
from ..prompts import SYSTEM_PROMPT
from ..runner import run_dialogues
from ..scenarios import Dialogue
from ..scoring import summarize

SITES = ("resid_post", "mlp_in", "mlp_out", "attn_out")
MATRICES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
MAX_TEXTS = 64


class ProbeBudgetExceeded(Exception):
    pass


class ProbeEnv:
    def __init__(self, model, tok, dev_dialogues: list[Dialogue], *, rule: str, project: str,
                 budget_fraction: float, max_probe_batches: int = 25, eval_size: int = 48,
                 seed: int = 0, batch_size: int = 16):
        if not 0 < budget_fraction <= 1 or not 0 <= max_probe_batches <= 25 or batch_size < 1:
            raise ValueError("expected a positive weight fraction <= 1, 0–25 probes and positive batch_size")
        self.model, self.tok = model, tok
        self.dev = dev_dialogues
        self.rule, self.project = rule, project
        self.budget_fraction = budget_fraction
        self.budget = budget_entries(model, budget_fraction)
        self.max_probe_batches = max_probe_batches
        self.probe_batches = 0
        self.vectors: dict[str, torch.Tensor] = {}
        self.edit: WeightEdit | None = None
        self.edit_spec = None
        self.history: list[dict] = []
        self.batch_size = batch_size
        rng = random.Random(seed)
        # default evaluation subset: balanced over conditions
        by_c = {}
        for d in dev_dialogues:
            by_c.setdefault(d.condition, []).append(d)
        self.eval_ids = []
        for c, ds in sorted(by_c.items()):
            self.eval_ids += [d.dialogue_id for d in rng.sample(ds, min(len(ds), eval_size // len(by_c)))]
        self.by_id = {d.dialogue_id: d for d in dev_dialogues}

    # ------------------------------------------------------------------ utils
    def _probe(self):
        if self.probe_batches >= self.max_probe_batches:
            raise ProbeBudgetExceeded(f"probe budget of {self.max_probe_batches} batches used up")
        self.probe_batches += 1

    def _layers(self):
        return self.model.model.layers

    def _param(self, name):
        return dict(self.model.named_parameters())[name]

    def _resolve_text(self, item) -> str:
        """item: raw string, {"user": str} (wrapped in chat template) or
        {"dialogue": id, "turn": k} (gold history up to that turn, generation prompt)."""
        if isinstance(item, str):
            return item
        if "dialogue" in item:
            d = self.by_id[item["dialogue"]]
            k = int(item.get("turn", 0))
            if not 0 <= k < len(d.turns):
                raise ValueError("turn must index a scored turn, starting at 0")
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
            for t in d.turns[:k]:
                msgs += [{"role": "user", "content": t.user}, {"role": "assistant", "content": t.gold}]
            msgs.append({"role": "user", "content": d.turns[k].user})
            return chat_text(self.tok, msgs)
        if "user" in item:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": item["user"]}]
            return chat_text(self.tok, msgs)
        raise ValueError(f"bad text item {item}")

    def _position(self, text, ids_offsets, position: str) -> int:
        if position == "last":
            return len(ids_offsets) - 1
        m = re.fullmatch(r"token:(.+)", position)
        if not m:
            raise ValueError("position must be 'last' or 'token:<substring>'")
        sub = m.group(1)
        at = text.rfind(sub)
        if at < 0:
            raise ValueError(f"substring {sub!r} not in text")
        end = at + len(sub)
        return max(i for i, (s, e) in enumerate(ids_offsets) if s < end and e > at)

    # ------------------------------------------------------------------ tools
    def model_info(self) -> dict:
        cfg = self.model.config
        return {
            "architecture": type(self.model).__name__,
            "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_attention_heads": cfg.num_attention_heads,
            "total_params": total_params(self.model),
            "edit_budget_entries": self.budget, "budget_fraction": self.budget_fraction,
            "editable_matrices": {m: list(self._param(f"model.layers.0.{m}.weight").shape) for m in MATRICES},
            "param_name_format": "model.layers.{L}.{matrix}.weight  (shape [out, in]; y = W x)",
            "capture_sites": {
                "resid_post": "residual stream after layer L (dim hidden_size)",
                "mlp_in": "input to mlp.down_proj at layer L (dim intermediate_size)",
                "mlp_out": "output of the MLP at layer L (dim hidden_size)",
                "attn_out": "output of self-attention at layer L (dim hidden_size)"},
            "probe_batches_used": self.probe_batches, "probe_batches_max": self.max_probe_batches,
        }

    @torch.no_grad()
    def read_weights(self, param: str, rows: list[int], cols: list[int]) -> dict:
        """Inspect a small submatrix of the currently edited model without a forward pass."""
        weight = self._param(param)
        if weight.ndim != 2 or not rows or not cols or len(rows) * len(cols) > 256:
            raise ValueError("select between 1 and 256 entries of a matrix")
        for indices, size in zip((rows, cols), weight.shape):
            if any(type(i) is not int or not 0 <= i < size for i in indices):
                raise ValueError(f"indices must be integers from 0 to {size - 1}")
        r = torch.tensor(rows, device=weight.device)
        c = torch.tensor(cols, device=weight.device)
        return {"param": param, "rows": rows, "cols": cols, "values": weight[r[:, None], c].float().tolist()}

    def list_dev(self, condition: str | None = None, limit: int = 4, offset: int = 0) -> list[dict]:
        ds = [d for d in self.dev if condition is None or d.condition == condition][offset:offset + limit]
        return [{"dialogue_id": d.dialogue_id, "condition": d.condition, "project": d.project,
                 "turns": [{"user": t.user, "gold": t.gold, "gold_if_no_rule": t.gold_no_rule}
                           for t in d.turns]} for d in ds]

    def _text_batches(self, texts, *, offsets=False):
        if not 0 < len(texts) <= MAX_TEXTS:
            raise ValueError(f"expected 1 to {MAX_TEXTS} probe texts")
        for start in range(0, len(texts), self.batch_size):
            batch = [self._resolve_text(x) for x in texts[start:start + self.batch_size]]
            enc = self.tok(batch, return_tensors="pt", padding=True, return_offsets_mapping=offsets,
                           add_special_tokens=False)
            mapping = enc.pop("offset_mapping").tolist() if offsets else None
            mask = enc.attention_mask
            if not mask.any(dim=1).all():
                raise ValueError("probe texts must contain tokens")
            # The last real token can differ across examples with right padding.
            last = (mask * torch.arange(mask.size(1))).max(dim=1).values
            yield batch, enc.to(self.model.device), mapping, last.to(self.model.device)

    @torch.no_grad()
    def capture(self, texts: list, layer: int, site: str, position: str = "last",
                name: str | None = None, reduce: str = "mean") -> dict:
        self._probe()
        if site not in SITES:
            raise ValueError(f"site must be one of {SITES}")
        L = self._layers()[layer]
        mod, kind = {"resid_post": (L, "out"), "mlp_in": (L.mlp.down_proj, "pre"),
                     "mlp_out": (L.mlp, "out"), "attn_out": (L.self_attn, "out")}[site]
        vecs = []
        for batch, enc, offsets, last in self._text_batches(texts, offsets=True):
            positions = last if position == "last" else [
                self._position(text, mapping, position) for text, mapping in zip(batch, offsets)]
            rows = torch.arange(len(batch), device=self.model.device)

            def collect(activation):
                vecs.append(activation[rows, positions].float().cpu())

            if kind == "pre":
                handle = mod.register_forward_pre_hook(lambda m, a: collect(a[0]))
            else:
                handle = mod.register_forward_hook(
                    lambda m, a, o: collect(o[0] if isinstance(o, tuple) else o))
            try:
                self.model.model(**enc, use_cache=False)
            finally:
                handle.remove()
        V = torch.cat(vecs)
        v = V.mean(0) if reduce == "mean" else V[0]
        name = name or f"cap_{site}_L{layer}_{len(self.vectors)}"
        self.vectors[name] = v
        return {"name": name, **self._vec_summary(v),
                "per_text_cosine_to_mean": [round(float(torch.cosine_similarity(x, v, 0)), 3) for x in V[:8]]}

    @torch.no_grad()
    def next_token_logprobs(self, texts: list, candidates: list[str]) -> list[dict]:
        """Log-probs of candidate continuations' first tokens at the generation position."""
        self._probe()
        cand_ids = {c: self.tok(c, add_special_tokens=False)["input_ids"][0] for c in candidates}
        out = []
        for batch, enc, _, last in self._text_batches(texts):
            hidden = self.model.model(**enc, use_cache=False).last_hidden_state
            # OLMo 2's LM head is the output projection of the final hidden state.
            rows = torch.arange(len(batch), device=hidden.device)
            logits = self.model.get_output_embeddings()(hidden[rows, last])
            for lp in logits.float().log_softmax(-1).cpu():
                top = lp.topk(min(5, lp.numel()))
                out.append({"logprobs": {c: round(float(lp[i]), 3) for c, i in cand_ids.items()},
                            "top5": [(self.tok.decode([int(i)]), round(float(v), 3))
                                     for v, i in zip(top.values, top.indices)]})
        return out

    def evaluate(self, dialogue_ids: list[str] | None = None, show_errors: int = 5) -> dict:
        """Run dev dialogues (model's own replies kept in history) with the current edit."""
        self._probe()
        ids = dialogue_ids or self.eval_ids
        ds = [self.by_id[i] for i in ids[:96]]
        recs = run_dialogues(self.model, self.tok, ds, batch_size=self.batch_size)
        summ = summarize(recs)
        errs = [{"dialogue_id": r["dialogue_id"], "condition": r["condition"], "turn": r["turn"],
                 "pred": r["pred"], "raw": r["raw"], "gold": r["gold"]}
                for r in recs if not r["correct"]][:show_errors]
        return {"summary": summ, "n_dialogues": len(ds), "errors": errs}

    # ---------------------------------------------------------- vector tools
    def _vec_summary(self, v):
        top = v.abs().topk(min(8, v.numel()))
        return {"dim": v.numel(), "norm": round(float(v.norm()), 4),
                "top_dims": [(int(i), round(float(v[i]), 4)) for i in top.indices]}

    @torch.no_grad()
    def token_direction(self, token: str, name: str | None = None, source: str = "unembed") -> dict:
        tid = self.tok(token, add_special_tokens=False)["input_ids"]
        if not tid or source not in ("embed", "unembed"):
            raise ValueError("expected a nonempty token and embed or unembed source")
        W = (self.model.lm_head.weight if source == "unembed" else self.model.model.embed_tokens.weight)
        v = W[tid].float().mean(0).detach().cpu()
        name = name or f"tok_{source}_{token.strip()}"
        self.vectors[name] = v
        return {"name": name, "token_ids": tid, **self._vec_summary(v)}

    @torch.no_grad()
    def vector_op(self, out: str, op: str, a: str, b: str | None = None, scalar: float = 1.0,
                  k: int | None = None) -> dict:
        A = self.vectors[a]
        B = self.vectors[b] if op in ("sub", "add", "project_out", "cosine") else None
        if op == "sub":
            v = A - B
        elif op == "add":
            v = A + B
        elif op == "scale":
            v = A * scalar
        elif op == "normalize":
            v = A / (A.norm() + 1e-8)
        elif op == "project_out":          # remove B's direction from A
            Bn = B / (B.norm() + 1e-8)
            v = A - (A @ Bn) * Bn
        elif op == "topk":                 # keep k largest-magnitude entries
            v = torch.zeros_like(A)
            idx = A.abs().topk(k).indices
            v[idx] = A[idx]
        elif op == "cosine":
            return {"cosine": float(torch.cosine_similarity(A, B, 0))}
        elif op == "map_through":           # B is a param name: v = W @ A  or  W^T @ A
            W = self._param(b).detach().cpu().float()
            v = W @ A if W.shape[1] == A.numel() else W.T @ A
        else:
            raise ValueError(f"unknown op {op}")
        self.vectors[out] = v
        return {"name": out, **self._vec_summary(v)}

    # ------------------------------------------------------------ edit tools
    def build_edit(self, spec: list[dict]) -> WeightEdit:
        """spec items:
          {"type":"rank1","param":P,"u":vec(out dim),"v":vec(in dim),"scale":s,
           "rows":k_rows|null,"cols":k_cols|null}
             -> dW[r,c] = scale*u[r]*v[c] on the top-|u| rows x top-|v| cols
          {"type":"entries","param":P,"entries":[[row,col,delta],...]}
        """
        if not spec:
            raise ValueError("an edit needs at least one operation")
        ops = []
        remaining = self.budget
        per_param = {}
        for it in spec:
            p = self._param(it["param"])
            out_d, in_d = p.shape
            if it["type"] == "rank1":
                u, v = self.vectors[it["u"]], self.vectors[it["v"]]
                if u.numel() != out_d or v.numel() != in_d:
                    raise ValueError(f"{it['param']} needs u[{out_d}] and v[{in_d}]")
                nr = out_d if it.get("rows") is None else it["rows"]
                nc = in_d if it.get("cols") is None else it["cols"]
                if type(nr) is not int or type(nc) is not int or not (0 < nr <= out_d and 0 < nc <= in_d):
                    raise ValueError("rows and cols must be positive integers within the matrix dimensions")
                if nr * nc > remaining:
                    raise ValueError(f"rank1 edit exceeds remaining budget of {remaining} entries")
                rows = u.abs().topk(nr).indices
                cols = v.abs().topk(nc).indices
                vals = float(it.get("scale", 1.0)) * torch.outer(u[rows], v[cols])
                flat = (rows[:, None] * in_d + cols[None, :]).reshape(-1)
            elif it["type"] == "entries":
                entries = it["entries"]
                if not 0 < len(entries) <= remaining:
                    raise ValueError(f"expected 1 to {remaining} entries within the edit budget")
                for row, col, delta in entries:
                    if (type(row) is not int or type(col) is not int
                            or not (0 <= row < out_d and 0 <= col < in_d)):
                        raise ValueError(f"invalid coordinate ({row}, {col}) for {list(p.shape)}")
                flat = torch.tensor([r * in_d + c for r, c, _ in entries])
                vals = torch.tensor([v for _, _, v in entries], dtype=torch.float32)
                if flat.unique().numel() != flat.numel():
                    raise ValueError(f"duplicate entries in {it['param']}")
            else:
                raise ValueError(f"unknown edit type {it['type']}")
            remaining -= flat.numel()
            per_param.setdefault(it["param"], []).append(flat)
            ops.append(EditOp(param=it["param"], kind="sparse", indices=flat, values=vals.reshape(-1)))
        for param, blocks in per_param.items():
            if len(blocks) > 1:
                flat = torch.cat(blocks)
                if flat.unique().numel() != flat.numel():
                    raise ValueError(f"overlapping entries in {param}")
        return WeightEdit(ops=ops)

    def apply_edit(self, spec: list[dict]) -> dict:
        try:
            new = self.build_edit(spec)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        n = new.n_entries(self.model)
        if self.edit is not None:
            self.edit.revert(self.model)
        try:
            new.apply(self.model, require_change=True)
        except Exception:
            if self.edit is not None:
                self.edit.apply(self.model)
            raise
        self.edit, self.edit_spec = new, spec
        return {"ok": True, "entries": n, "budget": self.budget, "l2_norm": round(new.l2_norm(), 4)}

    def revert_edit(self) -> dict:
        if self.edit is not None:
            self.edit.revert(self.model)
        self.edit, self.edit_spec = None, None
        return {"ok": True}

    # -------------------------------------------------------------- dispatch
    def call(self, name: str, args: dict):
        tools = {"model_info": self.model_info, "list_dev": self.list_dev, "capture": self.capture,
              "next_token_logprobs": self.next_token_logprobs, "evaluate": self.evaluate,
              "token_direction": self.token_direction, "vector_op": self.vector_op,
              "apply_edit": self.apply_edit, "revert_edit": self.revert_edit, "read_weights": self.read_weights}
        try:
            res = tools[name](**args)
        except ProbeBudgetExceeded as e:
            res = {"error": str(e)}
        except Exception as e:                       # report tool errors back to the agent
            res = {"error": f"{type(e).__name__}: {e}"}
        self.history.append({"tool": name, "args": args, "probe_batches": self.probe_batches})
        return res


def to_json(x) -> str:
    return json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
