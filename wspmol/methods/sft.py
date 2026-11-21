"""SFT (LoRA) baseline trained on the same labelled development dialogues WSP sees."""
from __future__ import annotations

import random

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from ..model import chat_text
from ..prompts import SYSTEM_PROMPT
from ..scenarios import Dialogue


def build_examples(tok, dialogues: list[Dialogue], max_len: int = 1536):
    """One example per scored turn: gold-answer history, loss only on the answer tokens."""
    ex = []
    for d in dialogues:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for t in d.turns:
            msgs.append({"role": "user", "content": t.user})
            prefix = chat_text(tok, msgs, add_generation_prompt=True)
            full = chat_text(tok, msgs + [{"role": "assistant", "content": t.gold}],
                             add_generation_prompt=False)
            p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
            if f_ids[: len(p_ids)] != p_ids or len(f_ids) > max_len:
                continue
            labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
            ex.append((f_ids, labels))
            msgs.append({"role": "assistant", "content": t.gold})
    return ex


def train_lora(model, tok, dialogues, out_dir, *, r=16, alpha=32, lr=2e-4, epochs=2,
               batch_size=8, grad_accum=1, seed=0, target_modules=None, log=print):
    torch.manual_seed(seed)
    random.seed(seed)
    target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, target_modules=target_modules,
                     task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    pm.print_trainable_parameters()
    examples = build_examples(tok, dialogues)
    log(f"SFT examples: {len(examples)}")
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=lr)
    steps = (len(examples) + batch_size - 1) // batch_size * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / max(1, steps)))
    pm.train()
    step = 0
    pad = tok.pad_token_id
    for ep in range(epochs):
        random.shuffle(examples)
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            L = max(len(x) for x, _ in batch)
            ids = torch.full((len(batch), L), pad)
            lab = torch.full((len(batch), L), -100)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, (x, y) in enumerate(batch):          # right padding for training
                ids[j, :len(x)] = torch.tensor(x)
                lab[j, :len(y)] = torch.tensor(y)
                att[j, :len(x)] = 1
            dev = pm.device
            loss = pm(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev)).loss
            (loss / grad_accum).backward()
            if (i // batch_size + 1) % grad_accum == 0:
                opt.step(); sched.step(); opt.zero_grad()
            if step % 20 == 0:
                log(f"epoch {ep} step {step}/{steps} loss {loss.item():.4f}")
            step += 1
    pm.eval()
    pm.save_pretrained(out_dir)
    return pm


def load_lora(model, adapter_dir, merge=True):
    pm = PeftModel.from_pretrained(model, adapter_dir)
    pm.eval()
    return pm.merge_and_unload() if merge else pm
