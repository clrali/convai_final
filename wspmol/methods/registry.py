"""Uniform access to every compared method."""
from __future__ import annotations

from pathlib import Path

from ..edits import WeightEdit, random_matched_edit

MODEL_METHODS = ["unedited", "prompting", "wsp", "wsp_attempt", "random_edit", "sft_lora", "rome"]
ALL_METHODS = MODEL_METHODS + ["oracle"]


def prepare(method: str, model, tok, rule: str, artifacts: Path, edit_path: str | None = None,
            seed: int = 0, *, expected_sha256: str | None = None):
    """Mutates `model` as required; returns (model, use_memory, info).

    Only `prompting` reads the saved-rule store (oracle is handled separately);
    WSP / SFT / ROME keep their original adaptation in every condition.
    """
    info = {"method": method, "rule": rule}
    if method == "unedited":
        return model, False, info
    if method == "prompting":
        return model, True, info
    if method in ("wsp", "wsp_attempt", "random_edit"):
        path = Path(edit_path) if edit_path else artifacts / "edits" / f"wsp_{rule}_primary.pt"
        edit = WeightEdit.load(path, expected_sha256=expected_sha256)
        info["edit"] = str(path)
        if method == "random_edit":
            edit = random_matched_edit(model, edit, seed=seed)
        edit.apply(model)
        info["entries"] = edit.n_entries(model)
        info["l2"] = edit.l2_norm()
        return model, False, info
    if method == "rome":
        path = Path(edit_path) if edit_path else artifacts / "edits" / f"rome_{rule}.pt"
        edit = WeightEdit.load(path)
        edit.apply(model)
        info["edit"] = str(path)
        return model, False, info
    if method == "sft_lora":
        from .sft import load_lora
        path = Path(edit_path) if edit_path else artifacts / "sft" / f"lora_{rule}"
        info["adapter"] = str(path)
        return load_lora(model, path, merge=True), False, info
    raise ValueError(method)
