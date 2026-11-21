"""Exercise LoRA gradients and accumulation on a tiny offline OLMo-2 model."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wspmol.methods import sft


@pytest.mark.parametrize("checkpointing", [False, True])
def test_accumulation_matches_large_batch(monkeypatch, tmp_path, checkpointing):
    # Unequal target lengths exercise token weighting; three examples also
    # exercise a partial final window, which must be applied every epoch.
    examples = [([1, 2, 3], [-100, -100, 3]),
                ([1, 4, 5, 6], [-100, -100, 5, 6]),
                ([1, 7, 8], [-100, -100, 8])]
    monkeypatch.setattr(sft, "build_examples", lambda *args: list(examples))
    config_cls = sft.LoraConfig

    def no_dropout(**kwargs):
        kwargs["lora_dropout"] = 0.0
        return config_cls(**kwargs)

    monkeypatch.setattr(sft, "LoraConfig", no_dropout)
    torch.manual_seed(42)
    model = Olmo2ForCausalLM(Olmo2Config(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, pad_token_id=0,
    ))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    lrs = []
    real_step = torch.optim.AdamW.step

    def record_step(opt, *args, **kwargs):
        lrs.append(opt.param_groups[0]["lr"])
        return real_step(opt, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", record_step)
    common = dict(r=2, alpha=4, epochs=2, lr=1e-3, log=lambda *args: None,
                  gradient_checkpointing=checkpointing)
    large = sft.train_lora(copy.deepcopy(model), SimpleNamespace(pad_token_id=0), [],
                           tmp_path / "large", batch_size=2, grad_accum=1, **common)
    micro = sft.train_lora(model, SimpleNamespace(pad_token_id=0), [],
                           tmp_path / "micro", batch_size=1, grad_accum=2, **common)
    expected_lrs = [1e-3, 7.5e-4, 5e-4, 2.5e-4]
    assert lrs == pytest.approx(expected_lrs * 2)
    for (name, a), (_, b) in zip(large.named_parameters(), micro.named_parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        if "lora_B" in name:
            assert torch.count_nonzero(b).item() > 0
    for name, value in before.items():
        base_name = "base_model.model." + name
        if base_name not in dict(micro.named_parameters()):
            base_name = base_name.replace(".weight", ".base_layer.weight")
        torch.testing.assert_close(dict(micro.named_parameters())[base_name], value)
    assert not micro.is_gradient_checkpointing
    assert micro.config.use_cache
    assert (tmp_path / "micro" / "adapter_config.json").is_file()
