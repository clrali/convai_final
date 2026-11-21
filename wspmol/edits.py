"""Reversible weight edits.

Two representations:
  sparse  : flat indices + additive deltas into one parameter (WSP, random ablation)
  lowrank : dense additive delta U @ V^T into one parameter (ROME)
Applying records the exact original values so revert() restores weights bitwise.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch


@dataclass
class EditOp:
    param: str
    kind: str                         # "sparse" | "lowrank"
    indices: torch.Tensor | None = None   # int64 flat indices (sparse)
    values: torch.Tensor | None = None    # float32 deltas (sparse)
    U: torch.Tensor | None = None         # [out, r] (lowrank)
    V: torch.Tensor | None = None         # [in, r]  (lowrank)

    def n_entries(self, shape) -> int:
        if self.kind == "sparse":
            return int(self.indices.numel())
        return int(shape[0] * shape[1])

    def to_dict(self):
        d = {"param": self.param, "kind": self.kind}
        for k in ("indices", "values", "U", "V"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v.cpu()
        return d


@dataclass
class WeightEdit:
    ops: list[EditOp] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    _saved: list = field(default_factory=list, repr=False)

    # ---------- bookkeeping ----------
    def n_entries(self, model) -> int:
        params = dict(model.named_parameters())
        return sum(op.n_entries(params[op.param].shape) for op in self.ops)

    def l2_norm(self) -> float:
        tot = 0.0
        for op in self.ops:
            if op.kind == "sparse":
                tot += float(op.values.float().pow(2).sum())
            else:
                tot += float((op.U.float() @ op.V.float().T).pow(2).sum())
        return tot ** 0.5

    def save(self, path):
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ops": [op.to_dict() for op in self.ops], "meta": self.meta}, path)

    @staticmethod
    def load(path) -> "WeightEdit":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return WeightEdit(ops=[EditOp(**o) for o in d["ops"]], meta=d.get("meta", {}))

    # ---------- apply / revert ----------
    @torch.no_grad()
    def apply(self, model):
        if self._saved:
            raise RuntimeError("edit already applied")
        params = dict(model.named_parameters())
        for op in self.ops:
            p = params[op.param]
            if op.kind == "sparse":
                idx = op.indices.to(p.device)
                flat = p.data.view(-1)
                orig = flat[idx].clone()
                flat[idx] = (orig.float() + op.values.to(p.device).float()).to(p.dtype)
                self._saved.append((op.param, "sparse", idx, orig))
            else:
                orig = p.data.clone().cpu()
                delta = op.U.to(p.device).float() @ op.V.to(p.device).float().T
                p.data.copy_((p.data.float() + delta).to(p.dtype))
                self._saved.append((op.param, "dense", None, orig))
        return self

    @torch.no_grad()
    def revert(self, model):
        params = dict(model.named_parameters())
        for name, kind, idx, orig in reversed(self._saved):
            p = params[name]
            if kind == "sparse":
                p.data.view(-1)[idx] = orig
            else:
                p.data.copy_(orig.to(p.device))
        self._saved.clear()
        return self

    @property
    def applied(self) -> bool:
        return bool(self._saved)


def total_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def budget_entries(model, fraction: float) -> int:
    return int(total_params(model) * fraction)


@torch.no_grad()
def weights_hash(model) -> str:
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters(), key=lambda x: x[0]):
        h.update(n.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def random_matched_edit(model, edit: WeightEdit, seed: int = 0) -> WeightEdit:
    """Size- and norm-matched random edit: same params, same #entries and L2 norm per op."""
    g = torch.Generator().manual_seed(seed)
    params = dict(model.named_parameters())
    ops = []
    for op in edit.ops:
        p = params[op.param]
        if op.kind == "sparse":
            n = op.indices.numel()
            target = op.values.float().norm()
        else:
            delta = op.U.float() @ op.V.float().T
            n = delta.numel() if delta.numel() < 2_000_000 else int((delta != 0).sum())
            target = delta.norm()
        idx = torch.randperm(p.numel(), generator=g)[:n]
        vals = torch.randn(n, generator=g)
        vals = vals / vals.norm() * target
        ops.append(EditOp(param=op.param, kind="sparse", indices=idx, values=vals))
    return WeightEdit(ops=ops, meta={"source": "random_matched", "seed": seed,
                                     "matched_to": edit.meta.get("name", "")})
