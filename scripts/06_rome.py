"""ROME: store the project rule as a fact, run the factual rewrite check, save the edit."""
from common import ART, base_parser, data_path, dump, logger

from wspmol.methods.rome import rewrite_check, rome_edit
from wspmol.model import load_model
from wspmol.scenarios import load


def cov_corpus(n_texts):
    """Generic text for the key covariance: wikitext-103 if available, else dev dialogue text."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        out = []
        for r in ds:
            if len(r["text"]) > 200:
                out.append(r["text"])
            if len(out) >= n_texts:
                return out, "wikitext-103"
    except Exception as e:
        print("wikitext unavailable:", e)
    texts = []
    for rule in ["chlorine", "mass"]:
        for d in load(data_path(rule, "dev_train")):
            texts += [t.user for t in d.turns]
    return texts[:n_texts], "dev dialogue text (fallback)"


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--rule", required=True, choices=["chlorine", "mass"])
    ap.add_argument("--layer", type=int, default=None, help="default: num_layers // 2 - 1")
    ap.add_argument("--cov-texts", type=int, default=2000)
    ap.add_argument("--cov-tokens", type=int, default=100_000)
    a = ap.parse_args()
    import torch
    log = logger(ART / "logs" / f"rome_{a.rule}.txt")
    model, tok = load_model(a.model, a.device, dtype=torch.float32)  # ROME solve in fp32
    layer = a.layer if a.layer is not None else model.config.num_hidden_layers // 2 - 1
    texts, src = cov_corpus(a.cov_texts)
    log(f"layer {layer}; covariance corpus: {src} ({len(texts)} texts)")
    before = rewrite_check(model, tok, a.rule)
    edit, info = rome_edit(model, tok, a.rule, layer, texts, cov_tokens=a.cov_tokens, log=log)
    edit.apply(model)
    after = rewrite_check(model, tok, a.rule)
    edit.revert(model)
    log(f"rewrite check before: {before}")
    log(f"rewrite check after:  {after}")
    passed = after["efficacy_pass"]
    edit.meta.update({"rewrite_check": after, "cov_source": src, **info})
    edit.save(ART / "edits" / f"rome_{a.rule}.pt")
    dump({"rule": a.rule, "layer": layer, "before": before, "after": after, "passed": passed,
          "cov_source": src}, ART / "edits" / f"rome_{a.rule}.json")
    if not passed:
        log("ROME FAILED the factual rewrite check - report failure; skip its dialogue evaluation")


if __name__ == "__main__":
    main()
