"""SFT (LoRA) on WSP's development examples (dev_train dialogues)."""
import argparse

from common import ART, base_parser, data_path, dump, logger

from wspmol.methods.sft import train_lora
from wspmol.model import load_model
from wspmol.scenarios import load


def main():
    ap = base_parser(__doc__)
    ap.set_defaults(batch_size=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rule", required=True, choices=["chlorine", "mass"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    a = ap.parse_args()
    out = ART / "sft" / f"lora_{a.rule}"
    log = logger(ART / "logs" / f"sft_{a.rule}.txt")
    # load_model uses bf16 base weights on CUDA; only LoRA adapters are trained.
    model, tok = load_model(a.model, a.device)
    dev = load(data_path(a.rule, "dev_train"))
    train_lora(model, tok, dev, out, r=a.r, alpha=2 * a.r, lr=a.lr, epochs=a.epochs,
               batch_size=a.batch_size, grad_accum=a.grad_accum,
               gradient_checkpointing=a.gradient_checkpointing, seed=a.seed, log=log)
    dump({"rule": a.rule, "n_dialogues": len(dev), "epochs": a.epochs, "lr": a.lr, "r": a.r,
          "batch_size": a.batch_size, "grad_accum": a.grad_accum,
          "gradient_checkpointing": a.gradient_checkpointing}, out / "train_meta.json")
    log("saved", out)


if __name__ == "__main__":
    main()
