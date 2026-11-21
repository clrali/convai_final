"""Create a tiny random OLMo-2-architecture model + BPE tokenizer for offline pipeline tests."""
import os
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import Olmo2Config, Olmo2ForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wspmol.scenarios import load  # noqa: E402

CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}"
    "{% if m['role'] == 'system' %}<|system|>\n{{ m['content'] }}\n"
    "{% elif m['role'] == 'user' %}<|user|>\n{{ m['content'] }}\n"
    "{% else %}<|assistant|>\n{{ m['content'] }}{{ eos_token }}\n{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def main(out=None):
    out = out or Path(os.environ.get("WSPMOL_RUN_DIR", ROOT)) / "artifacts" / "tiny_model"
    texts = []
    for f in Path(os.environ.get("WSPMOL_DATA_DIR", ROOT / "data")).glob("*_dev_train.jsonl"):
        for d in load(f)[:200]:
            texts += [t.user for t in d.turns]
    texts += ["The capital of France is Paris.", "Write a haiku about rain."] * 10
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    special = ["<|endoftext|>", "<pad>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>"]
    tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=1200, special_tokens=special,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<|endoftext|>", eos_token="<|endoftext|>",
                                  pad_token="<pad>", unk_token="<unk>")
    tok.chat_template = CHAT_TEMPLATE
    cfg = Olmo2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=2048,
                      pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    model = Olmo2ForCausalLM(cfg)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print("saved tiny model to", out, sum(p.numel() for p in model.parameters()), "params")


if __name__ == "__main__":
    main()
