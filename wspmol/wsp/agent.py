"""LLM agent loop (OpenAI-compatible tool calling) that writes a weight edit via ProbeEnv."""
from __future__ import annotations

import json
import time

from ..rules import RULES
from .probes import ProbeEnv, to_json

TOOLS = [
    {"name": "model_info", "description": "Model structure, parameter shapes, edit budget, probe usage. No probe cost.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "list_dev", "description": "Show labelled development dialogues (user turns and gold answers). No probe cost.",
     "parameters": {"type": "object", "properties": {
         "condition": {"type": "string", "enum": ["C1", "C2", "C3", "C4"]},
         "limit": {"type": "integer"}, "offset": {"type": "integer"}}}},
    {"name": "capture", "description": "Forward probe (1 batch): mean activation over texts at a layer/site/position; stored as a named vector.",
     "parameters": {"type": "object", "required": ["texts", "layer", "site"], "properties": {
         "texts": {"type": "array", "description": "Each: a raw string, {\"user\": str} (wrapped in the chat template with the task system prompt), or {\"dialogue\": id, \"turn\": k}.",
                   "items": {}},
         "layer": {"type": "integer"}, "site": {"type": "string", "enum": ["resid_post", "mlp_in", "mlp_out", "attn_out"]},
         "position": {"type": "string", "description": "'last' or 'token:<substring>' (last token of the last occurrence)"},
         "name": {"type": "string"}}}},
    {"name": "next_token_logprobs", "description": "Forward probe (1 batch): log-probs of candidate first tokens at the answer position, plus top-5.",
     "parameters": {"type": "object", "required": ["texts", "candidates"], "properties": {
         "texts": {"type": "array", "items": {}}, "candidates": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "evaluate", "description": "Forward probe (1 batch): run dev dialogues end-to-end with the current edit; returns turn accuracy, dialogue success, over-application per condition, and some errors.",
     "parameters": {"type": "object", "properties": {"dialogue_ids": {"type": "array", "items": {"type": "string"}}}}},
    {"name": "token_direction", "description": "Store a token's unembedding (or embedding) row as a vector. No probe cost.",
     "parameters": {"type": "object", "required": ["token"], "properties": {
         "token": {"type": "string"}, "name": {"type": "string"}, "source": {"type": "string", "enum": ["unembed", "embed"]}}}},
    {"name": "vector_op", "description": "Vector algebra on stored vectors: sub, add, scale, normalize, project_out, topk, cosine, map_through (b = parameter name; multiplies by W or W^T). No probe cost.",
     "parameters": {"type": "object", "required": ["out", "op", "a"], "properties": {
         "out": {"type": "string"}, "op": {"type": "string"}, "a": {"type": "string"}, "b": {"type": "string"},
         "scalar": {"type": "number"}, "k": {"type": "integer"}}}},
    {"name": "apply_edit", "description": "Replace the current tentative edit. Items: {type:'rank1', param, u, v, scale, rows, cols} adds scale*u[r]v[c] on the top-|u| rows x top-|v| cols; or {type:'entries', param, entries:[[row,col,delta],...]}. Total touched entries must fit the budget. Fully reversible. No probe cost.",
     "parameters": {"type": "object", "required": ["spec"], "properties": {"spec": {"type": "array", "items": {"type": "object"}}}}},
    {"name": "revert_edit", "description": "Remove the current edit (restores weights exactly).",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "finish", "description": "Submit the currently applied edit as this attempt's result.",
     "parameters": {"type": "object", "properties": {"notes": {"type": "string"}}}},
]


def system_prompt(env: ProbeEnv) -> str:
    rule = RULES[env.rule]
    return f"""You are performing weight-space programming on a language model (OLMo 2 1B Instruct).
You will directly edit its weights, using only forward probes (no gradients, no training).

Goal. The model is a molecular design assistant. For project "{env.project}" the user set the rule:
  "{rule.text()}"
Write a weight edit so that the model applies this rule in later sessions for project {env.project}
even when the rule is NOT restated (condition C2), while:
  - leaving other projects unaffected (C3: different project ID, no rule; the model must pick the
    closest candidate overall and must NOT apply {env.project}'s rule),
  - still following the rule when it is stated in context (C1),
  - still obeying an explicit in-conversation revision of the rule (C4).
Answers are a candidate letter or NONE.

Constraints. The edit may touch at most {env.budget:,} weight entries
({env.budget_fraction:.2%} of all weights). You have at most {env.max_probe_batches} probe batches
(each capture / next_token_logprobs / evaluate call is one batch). Edits are reversible: apply_edit
replaces any previous edit. When done, call finish; the edit applied at that moment is submitted
(if you run out of probes, the current edit is submitted). Use the labelled development dialogues to
form and test hypotheses. Be economical with probes."""


def run_agent(env: ProbeEnv, *, model: str, client=None, max_steps: int = 80,
              log=print, base_url: str | None = None) -> dict:
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=base_url) if base_url else OpenAI()
    tools = [{"type": "function", "function": t} for t in TOOLS]
    msgs = [{"role": "system", "content": system_prompt(env)},
            {"role": "user", "content": "Begin. Start with model_info and a look at the development dialogues."}]
    usage = {"input_tokens": 0, "output_tokens": 0}
    transcript = []
    notes, finished = "", False
    for step in range(max_steps):
        resp = client.chat.completions.create(model=model, messages=msgs, tools=tools)
        if getattr(resp, "usage", None):
            usage["input_tokens"] += resp.usage.prompt_tokens or 0
            usage["output_tokens"] += resp.usage.completion_tokens or 0
        msg = resp.choices[0].message
        msgs.append(msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else msg)
        calls = msg.tool_calls or []
        if msg.content:
            transcript.append({"step": step, "assistant": msg.content})
            log(f"[agent] {msg.content[:300]}")
        if not calls:
            msgs.append({"role": "user", "content": "Continue using the tools, or call finish."})
            continue
        for c in calls:
            args = json.loads(c.function.arguments or "{}")
            if c.function.name == "finish":
                notes, finished = args.get("notes", ""), True
                result = {"ok": True}
            else:
                result = env.call(c.function.name, args)
            log(f"[tool] {c.function.name}({to_json(args)[:200]}) -> {to_json(result)[:300]}")
            transcript.append({"step": step, "tool": c.function.name, "args": args, "result": result})
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": to_json(result)})
        if finished:
            break
        if env.probe_batches >= env.max_probe_batches:
            msgs.append({"role": "user", "content": "Probe budget exhausted. Make any final apply_edit "
                         "(no probes) and call finish."})
    return {"finished": finished, "notes": notes, "usage": usage, "probe_batches": env.probe_batches,
            "steps": step + 1, "edit_spec": env.edit_spec, "transcript": transcript,
            "tool_history": env.history}


def cost_usd(usage, in_per_m=2.0, out_per_m=10.0) -> float:
    return usage["input_tokens"] / 1e6 * in_per_m + usage["output_tokens"] / 1e6 * out_per_m
