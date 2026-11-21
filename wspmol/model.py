"""Model loading and batched multi-turn chat generation."""
from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "allenai/OLMo-2-0425-1B-Instruct"


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name: str = DEFAULT_MODEL, device: str | None = None, dtype=None):
    device = device or pick_device()
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
    model.eval()
    return model, tok


def chat_text(tok, messages, add_generation_prompt=True) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


@torch.no_grad()
def generate(model, tok, conversations: list[list[dict]], max_new_tokens: int = 8,
             batch_size: int = 16) -> list[str]:
    """Greedy-decode a short reply for each conversation."""
    outs = []
    for i in range(0, len(conversations), batch_size):
        texts = [chat_text(tok, c) for c in conversations[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True).strip())
    return outs


_LABEL_RE = re.compile(r"\b([A-E])\b")


def parse_answer(text: str, labels: list[str]) -> str:
    """Return a candidate label, 'NONE', or 'INVALID'."""
    t = text.strip()
    up = t.upper()
    m = _LABEL_RE.search(t)
    none_pos = up.find("NONE")
    if none_pos != -1 and (m is None or none_pos <= m.start()):
        return "NONE"
    if m and m.group(1) in labels:
        return m.group(1)
    return "INVALID"
