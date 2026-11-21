"""One WSP edit attempt: the agent LLM writes an edit using forward probes only."""
import hashlib
import json
import os

from common import ART, base_parser, data_path, dump, logger

from wspmol.model import load_model
from wspmol.scenarios import EDIT_PROJECTS, load
from wspmol.wsp.frozen import model_id
from wspmol.wsp.agent import cost_usd, run_agent
from wspmol.wsp.probes import ProbeEnv


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--rule", required=True, choices=["chlorine", "mass"])
    ap.add_argument("--budget", type=float, required=True, help="fraction of weights, e.g. 0.001 or 0.0001")
    ap.add_argument("--attempt", type=int, required=True)
    ap.add_argument("--agent-model", default=os.environ.get("AGENT_MODEL", "gpt-6.1-sol"), help="API model id of the editing LLM")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (e.g. for Astra)")
    ap.add_argument("--agent-api", choices=["responses", "chat"], default="responses")
    ap.add_argument("--max-probes", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--mock-agent", action="store_true", help="scripted agent (pipeline test only)")
    ap.add_argument("--resume", action="store_true", help="reuse an already completed matching attempt")
    a = ap.parse_args()
    if not 0 <= a.max_probes <= 25 or a.max_steps < 1 or not 0 < a.budget <= 1:
        ap.error("expected 0–25 probes, positive steps and a weight fraction in (0, 1]")
    name = f"wsp_{a.rule}_b{a.budget:g}_a{a.attempt}"
    if a.mock_agent:
        name = "mock_" + name
    out = ART / "edits"
    log = logger(ART / "logs" / f"{name}.txt")
    train_data = data_path(a.rule, "dev_train")
    metadata = {"name": name, "rule": a.rule, "budget": a.budget, "attempt": a.attempt,
                "target_model": a.model, "train_sha256": hashlib.sha256(train_data.read_bytes()).hexdigest(), "agent_model": "mock" if a.mock_agent else a.agent_model,
                "agent_api": "mock" if a.mock_agent else a.agent_api, "seed": a.seed + a.attempt,
                "max_probe_batches": a.max_probes, "max_steps": a.max_steps, "batch_size": a.batch_size}
    report = out / f"{name}.json"
    weights = out / f"{name}.pt"
    if a.resume and report.is_file():
        prev = json.loads(report.read_text())
        if "target_model" in prev:
            prev["target_model"] = model_id(prev["target_model"])
        if prev.get("stop_reason") != "api_error" and all(prev.get(k) == v for k, v in metadata.items()):
            if not prev["usable"] or (weights.is_file() and
                                      hashlib.sha256(weights.read_bytes()).hexdigest() == prev.get("sha256")):
                log(f"Reusing completed attempt {name}")
                return
    client = None
    if not a.mock_agent:
        from openai import OpenAI
        client = OpenAI(base_url=a.base_url)
    model, tok = load_model(a.model, a.device)
    dev = load(train_data)
    env = ProbeEnv(model, tok, dev, rule=a.rule, project=EDIT_PROJECTS[a.rule],
                   budget_fraction=a.budget, max_probe_batches=a.max_probes,
                   seed=a.seed + a.attempt, batch_size=a.batch_size)
    if a.mock_agent:
        from wspmol.wsp.mock_agent import MockClient
        client = MockClient(env)
    res = run_agent(env, model=a.agent_model, client=client, max_steps=a.max_steps, log=log,
                    base_url=a.base_url, api="responses" if a.mock_agent else a.agent_api)
    res.update(metadata, cost_usd=cost_usd(res["usage"]))

    if res["stop_reason"] == "api_error":
        dump({**res, "usable": False}, report)
        raise SystemExit(f"Editing API failed: {res['api_error']}. Partial transcript saved to {report}")
    if env.edit is None or (not res["finished"] and env.probe_batches < env.max_probe_batches):
        log(f"No submitted edit (stopped at {res['stop_reason']}, probes={env.probe_batches})")
        dump({**res, "usable": False}, report)
        return
    env.edit.meta.update(metadata, spec=res["edit_spec"])
    env.edit.save(weights)
    res["sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    res.update({"usable": True, "entries": env.edit.n_entries(model), "l2": env.edit.l2_norm()})
    dump(res, report)
    log(f"saved {name}: entries={res['entries']} probes={res['probe_batches']} cost=${res['cost_usd']:.2f}")


if __name__ == "__main__":
    main()
