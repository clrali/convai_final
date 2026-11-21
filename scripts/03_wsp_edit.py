"""One WSP edit attempt: the agent LLM writes an edit using forward probes only."""
import json

from common import ART, base_parser, data_path, dump, logger

from wspmol.model import load_model
from wspmol.scenarios import EDIT_PROJECTS, load
from wspmol.wsp.agent import cost_usd, run_agent
from wspmol.wsp.probes import ProbeEnv


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--rule", required=True, choices=["chlorine", "mass"])
    ap.add_argument("--budget", type=float, required=True, help="fraction of weights, e.g. 0.001 or 0.0001")
    ap.add_argument("--attempt", type=int, required=True)
    ap.add_argument("--agent-model", default="gpt-6.1-sol", help="API model id of the editing LLM")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (e.g. for Astra)")
    ap.add_argument("--max-probes", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--mock-agent", action="store_true", help="scripted agent (pipeline test only)")
    a = ap.parse_args()
    name = f"wsp_{a.rule}_b{a.budget:g}_a{a.attempt}"
    out = ART / "edits"
    log = logger(ART / "logs" / f"{name}.txt")
    model, tok = load_model(a.model, a.device)
    dev = load(data_path(a.rule, "dev_train"))
    env = ProbeEnv(model, tok, dev, rule=a.rule, project=EDIT_PROJECTS[a.rule],
                   budget_fraction=a.budget, max_probe_batches=a.max_probes,
                   seed=a.seed + a.attempt, batch_size=a.batch_size)
    client = None
    if a.mock_agent:
        from wspmol.wsp.mock_agent import MockClient
        client = MockClient(env)
    res = run_agent(env, model=a.agent_model, client=client, max_steps=a.max_steps, log=log,
                    base_url=a.base_url)
    if env.edit is None:
        log("NO USABLE EDIT: agent submitted no edit")
        dump({**res, "usable": False}, out / f"{name}.json")
        return
    env.edit.meta.update({"name": name, "rule": a.rule, "budget": a.budget, "attempt": a.attempt,
                          "agent_model": a.agent_model, "spec": res["edit_spec"]})
    env.edit.save(out / f"{name}.pt")
    res.update({"usable": True, "entries": env.edit.n_entries(model), "l2": env.edit.l2_norm(),
                "cost_usd": cost_usd(res["usage"])})
    dump(res, out / f"{name}.json")
    log(f"saved {name}: entries={res['entries']} probes={res['probe_batches']} cost=${res['cost_usd']:.2f}")


if __name__ == "__main__":
    main()
