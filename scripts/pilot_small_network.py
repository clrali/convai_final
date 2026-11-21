"""Pilot GPT weight editing on a published XOR repair or the original quadrant task."""
import os

import torch
import torch.nn as nn
from common import ART, base_parser, dump, logger

from wspmol.wsp.agent import cost_usd, run_tool_loop

XOR_SOURCE = "https://github.com/ttm/weight-space-programming/blob/main/fixtures/xor2_corrupted.json"
TOOLS = [
    {"name": "get_weights", "description": "Read all current weights as nested lists.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "evaluate", "description": "Score current weights on the pilot task.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "forward", "description": "Return logits and hidden activations for points [[x,y],...].",
     "parameters": {"type": "object", "required": ["points"], "properties": {
         "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}}}},
    {"name": "set_weights", "description": "Assign entries [[parameter, row, column or null for bias, value], ...].",
     "parameters": {"type": "object", "required": ["entries"], "properties": {
         "entries": {"type": "array", "items": {"type": "array", "items": {"anyOf": [
             {"type": "string"}, {"type": "number"}, {"type": "null"}]}}}}}},
    {"name": "finish", "description": "Submit the current weights.",
     "parameters": {"type": "object", "properties": {}}},
]


def xor_repair():
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1., -1.], [-1., 1.]]))
        net[0].bias.zero_()
        net[2].weight.fill_(1.)
        net[2].bias.fill_(-1.5)
    original = {k: v.clone().detach() for k, v in net.named_parameters()}
    points = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])

    def evaluate():
        predictions = net(points).flatten() > 0
        correct = predictions == (points[:, 0] != points[:, 1])
        changed = sum(int((v != original[k]).sum()) for k, v in net.named_parameters())
        return {"accuracy": float(correct.float().mean()), "changed_entries": changed,
                "pass": bool(correct.all()) and changed == 1}

    task = ("The network is Linear(2,2), ReLU, Linear(2,1). Class 1 means the output is strictly positive. "
            "Repair XOR on all four binary inputs. The submitted network must differ in exactly one scalar.")
    return net, task, evaluate


def quadrant_repair():
    def data(seed):
        X = torch.rand(4000, 2, generator=torch.Generator().manual_seed(seed)) * 2 - 1
        return X, (X[:, 0] > 0).long() * 2 + (X[:, 1] > 0).long()

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 4))
    X, y = data(0)
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    for _ in range(1500):
        opt.zero_grad()
        nn.functional.cross_entropy(net(X), y).backward()
        opt.step()
    Xt, yt = data(1)
    target = Xt[:, 0] > 0.5

    def evaluate():
        pred = net(Xt).argmax(1)
        hit = float((pred[target] == 0).float().mean())
        retain = float((pred[~target] == yt[~target]).float().mean())
        return {"target_region_class0": hit, "retained_elsewhere": retain, "pass": min(hit, retain) >= 0.9}

    task = ("Linear(2,8), ReLU, Linear(8,4) predicts quadrant 2*(x>0)+(y>0) on [-1,1]^2. "
            "Make all points with x > 0.5 predict class 0 while retaining accuracy elsewhere.")
    return net, task, evaluate


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--task", choices=["xor", "quadrant"], default="quadrant")
    ap.add_argument("--agent-model", default=os.environ.get("AGENT_MODEL", "gpt-6.1-sol"))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--agent-api", choices=["responses", "chat"], default="responses")
    ap.add_argument("--max-steps", type=int, default=40)
    a = ap.parse_args()
    log = logger(ART / "logs" / "pilot_small_network.txt")
    net, task, evaluate = xor_repair() if a.task == "xor" else quadrant_repair()
    params = dict(net.named_parameters())

    @torch.no_grad()
    def call(name, args):
        if name == "get_weights":
            return {k: v.tolist() for k, v in params.items()}
        if name == "evaluate":
            return evaluate()
        if name == "forward":
            points = torch.tensor(args["points"], dtype=torch.float32)
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("expected points [[x,y],...]")
            return {"hidden": net[:2](points).tolist(), "logits": net(points).tolist()}
        if name == "set_weights":
            proposed = {k: v.clone() for k, v in params.items()}
            for t, i, j, value in args["entries"]:
                weight = proposed[t]
                index = (i,) if j is None else (i, j)
                if len(index) != weight.ndim or any(type(n) is not int or not 0 <= n < size
                                                     for n, size in zip(index, weight.shape)):
                    raise ValueError(f"invalid index {index} for {t}")
                weight[index] = float(value)
                if not weight[index].isfinite():
                    raise ValueError("weight must be finite")
            net.load_state_dict(proposed)
            return {"ok": True}
        raise ValueError(f"unknown tool {name}")

    before = call("evaluate", {})
    log("before", before)
    from openai import OpenAI
    run = run_tool_loop(
        client=OpenAI(base_url=a.base_url), model=a.agent_model, api=a.agent_api, max_steps=a.max_steps,
        instructions="Edit weights directly using the forward-only tools. Inspect the network and call finish to submit.",
        task=task, tools=TOOLS, dispatch=call, log=log)
    res = call("evaluate", {})
    res["pass"] = res["pass"] and run["finished"]
    log("after", res)
    res.update(run, before=before, task=a.task, source=XOR_SOURCE if a.task == "xor" else "local quadrant task",
               weights=call("get_weights", {}), agent_model=a.agent_model, cost_usd=cost_usd(run["usage"]))
    dump(res, ART / "pilot_small_network.json")
    if run["api_error"]:
        raise SystemExit(f"Pilot API failed: {run['api_error']}")


if __name__ == "__main__":
    main()
