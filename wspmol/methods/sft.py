"""SFT (LoRA) baseline trained on the same labelled development dialogues WSP sees."""
from __future__ import annotations

import math
import random
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from ..model import chat_text
from ..prompts import SYSTEM_PROMPT
from ..scenarios import Dialogue


def build_examples(tok, dialogues: list[Dialogue], max_len: int = 2048):
    """One example per DIALOGUE, with loss on every gold answer in it.

    The earlier per-turn version re-encoded the whole history for each turn, so a 3-turn
    dialogue cost 1+2+3 = 6 units of text instead of 3. Supervising every answer position of a
    single sequence is the same objective and ~61% cheaper on this data.
    """
    ex = []
    for d in dialogues:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        spans, final = [], None
        for t in d.turns:
            msgs.append({"role": "user", "content": t.user})
            prefix = tok(chat_text(tok, msgs, add_generation_prompt=True),
                         add_special_tokens=False)["input_ids"]
            msgs.append({"role": "assistant", "content": t.gold})
            full = tok(chat_text(tok, msgs, add_generation_prompt=False),
                       add_special_tokens=False)["input_ids"]
            if full[: len(prefix)] != prefix:     # chat template not prefix-stable; skip dialogue
                final = None
                break
            spans.append((len(prefix), len(full)))
            final = full
        if final is None or not spans or len(final) > max_len:
            continue
        labels = [-100] * len(final)
        for a, b in spans:
            labels[a:b] = final[a:b]
        ex.append((final, labels))
    return ex


def _batches(examples, batch_size, rng):
    """Length-grouped batches: sorting within a shuffled window keeps padding small."""
    order = list(range(len(examples)))
    rng.shuffle(order)
    window = batch_size * 20
    out = []
    for i in range(0, len(order), window):
        chunk = sorted(order[i:i + window], key=lambda j: len(examples[j][0]))
        for k in range(0, len(chunk), batch_size):
            out.append([examples[j] for j in chunk[k:k + batch_size]])
    rng.shuffle(out)
    return out


def train_lora(model, tok, dialogues, out_dir, *, r=16, alpha=32, lr=2e-4, epochs=2,
               batch_size=4, grad_accum=2, seed=0, target_modules=None,
               gradient_checkpointing=False, max_len=2048, log=print):
    if min(batch_size, grad_accum, epochs) < 1:
        raise ValueError("batch_size, grad_accum and epochs must be positive")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, target_modules=target_modules,
                     task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    pm.print_trainable_parameters()
    previous_cache = pm.config.use_cache
    if gradient_checkpointing:
        # transformers>=5 renamed this; fall back for older versions
        (getattr(pm, "activation_checkpointing_enable", None) or pm.gradient_checkpointing_enable)()
        pm.enable_input_require_grads()
        pm.config.use_cache = False       # incompatible with checkpointing; silences the warning

    examples = build_examples(tok, dialogues, max_len)
    toks = sum(len(x) for x, _ in examples)
    sup = sum(sum(1 for v in y if v != -100) for _, y in examples)
    log(f"SFT examples: {len(examples)} dialogues ({len(dialogues)} given), "
        f"{toks} tokens, {sup} supervised")

    if not sup:
        raise ValueError("No supervised tokens in the SFT examples")
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=lr)
    effective_batch = batch_size * grad_accum
    opt_steps = epochs * math.ceil(len(examples) / effective_batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / opt_steps))
    log(f"batch_size={batch_size} grad_accum={grad_accum} -> {opt_steps} optimizer steps")

    pm.train()
    pad, step, t0 = tok.pad_token_id, 0, time.time()
    for ep in range(epochs):
        for window in _batches(examples, effective_batch, rng):
            # Causal LM cross entropy is averaged over supervised next tokens.
            # Normalize once per optimizer update, including a partial last window.
            counts = [sum(v != -100 for v in labels[1:]) for _, labels in window]
            total = sum(counts)
            mean_loss = torch.zeros((), device=pm.device)
            for start in range(0, len(window), batch_size):
                batch = window[start:start + batch_size]
                n_tokens = sum(counts[start:start + batch_size])
                if not n_tokens:
                    continue
                length = max(len(x) for x, _ in batch)
                ids = torch.full((len(batch), length), pad)
                lab = torch.full((len(batch), length), -100)
                att = torch.zeros_like(ids)
                for j, (x, y) in enumerate(batch):
                    ids[j, :len(x)] = torch.tensor(x)
                    lab[j, :len(y)] = torch.tensor(y)
                    att[j, :len(x)] = 1
                dev = pm.device
                loss = pm(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev)).loss
                weighted_loss = loss * (n_tokens / total)
                weighted_loss.backward()
                mean_loss += weighted_loss.detach()
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0 or step == 1:
                log(f"epoch {ep} opt-step {step}/{opt_steps} loss {mean_loss.item():.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)")
    if gradient_checkpointing:
        (getattr(pm, "activation_checkpointing_disable", None) or pm.gradient_checkpointing_disable)()
        pm.disable_input_require_grads()
        pm.config.use_cache = previous_cache
    pm.eval()
    pm.save_pretrained(out_dir)
    log(f"done in {time.time()-t0:.0f}s")
    return pm


def load_lora(model, adapter_dir, merge=True):
    pm = PeftModel.from_pretrained(model, adapter_dir)
    pm.eval()
    return pm.merge_and_unload() if merge else pm
