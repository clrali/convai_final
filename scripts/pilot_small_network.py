"""Pilot: can the agent LLM edit a small network's weights by forward probes alone?

A 2-layer MLP is trained to classify 2-D points into 4 quadrant classes. The agent must edit
its weights (no gradients) so that every point with x > 0.5 is classified as class 0, while
keeping accuracy elsewhere. Success: >= 90% on the edited region and >= 90% retained elsewhere.
"""
import json

import torch
import torch.nn as nn
from common import ART, base_parser, dump, logger

TOOLS = [
    {"type": "function", "function": {"name": "get_weights", "description": "All weights as nested lists.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "evaluate", "description": "Accuracy on target region and elsewhere for current weights.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "forward", "description": "Logits and hidden activations for given points [[x,y],...].",
                                      "parameters": {"type": "object", "required": ["points"], "properties": {"points": {"type": "array", "items": {}}}}}},
    {"type": "function", "function": {"name": "set_weights", "description": "Set entries: [[tensor_name, i, j_or_null, value], ...]",
                                      "parameters": {"type": "object", "required": ["entries"], "properties": {"entries": {"type": "array", "items": {}}}}}},
    {"type": "function", "function": {"name": "finish", "description": "Submit.", "parameters": {"type": "object", "properties": {}}}},
]


def data(n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 2, generator=g) * 2 - 1
    y = (X[:, 0] > 0).long() * 2 + (X[:, 1] > 0).long()
    return X, y


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--agent-model", default="gpt-6.1-sol")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-steps", type=int, default=40)
    a = ap.parse_args()
    log = logger(ART / "logs" / "pilot_small_network.txt")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 4))
    X, y = data()
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    for _ in range(1500):
        opt.zero_grad(); nn.functional.cross_entropy(net(X), y).backward(); opt.step()
    Xt, yt = data(seed=1)
    target = Xt[:, 0] > 0.5
    params = dict(net.named_parameters())

    def evaluate():
        with torch.no_grad():
            p = net(Xt).argmax(1)
        return {"target_region_class0": float((p[target] == 0).float().mean()),
                "retained_elsewhere": float((p[~target] == yt[~target]).float().mean())}

    def call(name, args):
        with torch.no_grad():
            if name == "get_weights":
                return {k: v.tolist() for k, v in params.items()}
            if name == "evaluate":
                return evaluate()
            if name == "forward":
                P = torch.tensor(args["points"], dtype=torch.float32)
                return {"hidden": torch.relu(net[0](P)).tolist(), "logits": net(P).tolist()}
            if name == "set_weights":
                for t, i, j, v in args["entries"]:
                    if j is None:
                        params[t][i] = v
                    else:
                        params[t][i, j] = v
                return {"ok": True}
        return {"ok": True}

    log("before", evaluate())
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url) if a.base_url else OpenAI()
    msgs = [{"role": "system", "content": "You edit neural network weights directly; no training. "
             "Network: Linear(2,8) -> ReLU -> Linear(8,4) named '0.weight','0.bias','2.weight','2.bias'. "
             "It classifies points in [-1,1]^2 into quadrant classes 2*(x>0)+(y>0). Edit it so all points "
             "with x > 0.5 are classified as class 0, keeping other predictions unchanged."},
            {"role": "user", "content": "Begin."}]
    for _ in range(a.max_steps):
        m = client.chat.completions.create(model=a.agent_model, messages=msgs, tools=TOOLS).choices[0].message
        msgs.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            msgs.append({"role": "user", "content": "Use the tools or call finish."})
            continue
        done = False
        for c in m.tool_calls:
            res = call(c.function.name, json.loads(c.function.arguments or "{}"))
            done |= c.function.name == "finish"
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(res)[:20000]})
        if done:
            break
    res = evaluate()
    res["pass"] = res["target_region_class0"] >= 0.9 and res["retained_elsewhere"] >= 0.9
    log("after", res)
    dump(res, ART / "pilot_small_network.json")


if __name__ == "__main__":
    main()
