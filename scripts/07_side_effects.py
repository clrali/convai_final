"""Rollback check + 200-item instruction-following slice for each adaptation vs. the unedited model."""
from common import ART, RESULTS, base_parser, dump, logger

from wspmol.edits import WeightEdit, random_matched_edit
from wspmol.model import load_model
from wspmol.side_effects import instruction_following, rollback_check

PROBES = ["Project Heron. My starting molecule is toluene.", "The capital of France is",
          "Write a haiku about rain.", "Project Juniper here. Starting molecule: phenol."]


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--rules", nargs="+", default=["chlorine", "mass"])
    ap.add_argument("--n-if", type=int, default=200)
    a = ap.parse_args()
    log = logger(ART / "logs" / "side_effects.txt")
    model, tok = load_model(a.model, a.device)
    out = {"unedited": {"if": instruction_following(model, tok, a.n_if, batch_size=a.batch_size)}}
    log(f"unedited IF accuracy {out['unedited']['if']['accuracy']:.3f}")
    for rule in a.rules:
        edits = {"wsp": ART / "edits" / f"wsp_{rule}_primary.pt",
                 "rome": ART / "edits" / f"rome_{rule}.pt"}
        for name, path in edits.items():
            if not path.exists():
                continue
            e = WeightEdit.load(path)
            variants = {name: e}
            if name == "wsp":
                variants["random_edit"] = random_matched_edit(model, e, seed=a.seed)
            for vname, ed in variants.items():
                rb = rollback_check(model, tok, ed, PROBES)
                ed.apply(model)
                ifr = instruction_following(model, tok, a.n_if, batch_size=a.batch_size)
                ed.revert(model)
                out[f"{vname}/{rule}"] = {"rollback": rb, "if": ifr}
                log(f"{vname}/{rule}: rollback {rb}  IF {ifr['accuracy']:.3f}")
        lora = ART / "sft" / f"lora_{rule}"
        if lora.exists():
            from peft import PeftModel
            pm = PeftModel.from_pretrained(model, lora)
            ifr = instruction_following(pm, tok, a.n_if, batch_size=a.batch_size)
            out[f"sft_lora/{rule}"] = {"if": ifr}
            log(f"sft_lora/{rule}: IF {ifr['accuracy']:.3f}")
            model = pm.unload()           # removes adapter, restores base
    dump(out, RESULTS / "side_effects.json")


if __name__ == "__main__":
    main()
