"""LLM agent loop (OpenAI-compatible tool calling) that writes a weight edit via ProbeEnv."""
from __future__ import annotations

import json

from openai import APIError

from ..rules import RULES
from .probes import ProbeEnv, to_json

TEXT_ITEM = {"anyOf": [
    {"type": "string"},
    {"type": "object", "properties": {"user": {"type": "string"}}, "required": ["user"]},
    {"type": "object", "properties": {"dialogue": {"type": "string"}, "turn": {"type": "integer"}},
     "required": ["dialogue"]},
]}
EDIT_ITEM = {"anyOf": [
    {"type": "object", "required": ["type", "param", "u", "v"], "properties": {
        "type": {"type": "string", "enum": ["rank1"]}, "param": {"type": "string"},
        "u": {"type": "string"}, "v": {"type": "string"}, "scale": {"type": "number"},
        "rows": {"type": "integer"}, "cols": {"type": "integer"}}},
    {"type": "object", "required": ["type", "param", "entries"], "properties": {
        "type": {"type": "string", "enum": ["entries"]}, "param": {"type": "string"},
        "entries": {"type": "array", "items": {"type": "array", "items": {"type": "number"},
                                               "minItems": 3, "maxItems": 3}}}},
]}

TOOLS = [
    {"name": "model_info", "description": "Model structure, parameter shapes, edit budget, probe usage. No probe cost.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "read_weights", "description": "Read a submatrix of current weights (up to 256 values). No probe cost.",
     "parameters": {"type": "object", "required": ["param", "rows", "cols"], "properties": {
         "param": {"type": "string"}, "rows": {"type": "array", "items": {"type": "integer"}},
         "cols": {"type": "array", "items": {"type": "integer"}}}}},
    {"name": "list_dev", "description": "Show labelled development dialogues (user turns and gold answers). No probe cost.",
     "parameters": {"type": "object", "properties": {
         "condition": {"type": "string", "enum": ["C1", "C2", "C3", "C4"]},
         "limit": {"type": "integer"}, "offset": {"type": "integer"}}}},
    {"name": "capture", "description": "Forward probe (1 batch): mean activation over texts at a layer/site/position; stored as a named vector.",
     "parameters": {"type": "object", "required": ["texts", "layer", "site"], "properties": {
         "texts": {"type": "array", "description": "Each: a raw string, {\"user\": str} (wrapped in the chat template with the task system prompt), or {\"dialogue\": id, \"turn\": k}.",
                   "items": TEXT_ITEM},
         "layer": {"type": "integer"}, "site": {"type": "string", "enum": ["resid_post", "mlp_in", "mlp_out", "attn_out"]},
         "position": {"type": "string", "description": "'last' or 'token:<substring>' (last token of the last occurrence)"},
         "name": {"type": "string"}}}},
    {"name": "next_token_logprobs", "description": "Forward probe (1 batch): log-probs of candidate first tokens at the answer position, plus top-5.",
     "parameters": {"type": "object", "required": ["texts", "candidates"], "properties": {
         "texts": {"type": "array", "items": TEXT_ITEM}, "candidates": {"type": "array", "items": {"type": "string"}}}}},
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
     "parameters": {"type": "object", "required": ["spec"], "properties": {"spec": {"type": "array", "items": EDIT_ITEM}}}},
    {"name": "revert_edit", "description": "Remove the current edit (restores weights exactly).",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "finish", "description": "Submit the currently applied edit as this attempt's result.",
     "parameters": {"type": "object", "properties": {"notes": {"type": "string"}}}},
]


def system_prompt(env: ProbeEnv) -> str:
    rule = RULES[env.rule]
    return f"""You are performing weight-space programming on {type(env.model).__name__}.
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
(you may still use edit and vector tools after the last probe). Use the labelled development dialogues to
form and test hypotheses. Be economical with probes."""


def run_tool_loop(*, client, model, instructions, task, tools, dispatch, max_steps,
                  api="responses", log=print):
    """Run ordered local tools and return an auditable record, including unsuccessful attempts."""
    if max_steps < 1 or api not in ("responses", "chat"):
        raise ValueError("expected positive max_steps and api='responses' or 'chat'")
    messages = [{"role": "system", "content": instructions}, {"role": "user", "content": task}]
    inputs = [{"role": "user", "content": task}]
    previous_id = None
    usage = {"input_tokens": 0, "output_tokens": 0}
    transcript = []
    notes = ""
    stop = "step_limit"
    error = None
    for step in range(1, max_steps + 1):
        try:
            if api == "responses":
                resp = client.responses.create(
                    model=model, instructions=instructions, input=inputs, previous_response_id=previous_id,
                    tools=[{"type": "function", **t, "strict": False} for t in tools])
                previous_id = resp.id
                content = resp.output_text
                calls = [(c.call_id, c.name, c.arguments) for c in resp.output if c.type == "function_call"]
                if resp.usage:
                    usage["input_tokens"] += resp.usage.input_tokens
                    usage["output_tokens"] += resp.usage.output_tokens
                inputs = []
            else:
                resp = client.chat.completions.create(
                    model=model, messages=messages, tools=[{"type": "function", "function": t} for t in tools])
                msg = resp.choices[0].message
                messages.append(msg.model_dump(exclude_none=True))
                content = msg.content
                calls = [(c.id, c.function.name, c.function.arguments) for c in msg.tool_calls or []]
                if resp.usage:
                    usage["input_tokens"] += resp.usage.prompt_tokens
                    usage["output_tokens"] += resp.usage.completion_tokens
        except APIError as exc:
            stop = "api_error"
            error = {"type": type(exc).__name__, "http_status": getattr(exc, "status_code", None),
                     "request_id": getattr(exc, "request_id", None)}
            log(f"API request failed: {error}")
            break
        if content:
            transcript.append({"step": step, "assistant": content})
            log(f"[agent] {content[:300]}")
        for call_id, name, raw in calls:
            args = raw
            try:
                args = json.loads(raw)
                if not isinstance(args, dict):
                    raise ValueError("tool arguments must be a JSON object")
                if name == "finish":
                    notes, stop = args.get("notes", ""), "finish"
                    result = {"ok": True}
                else:
                    result = dispatch(name, args)
            except (TypeError, ValueError, KeyError, IndexError) as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
            log(f"[tool] {name}({to_json(args)[:200]}) -> {to_json(result)[:300]}")
            transcript.append({"step": step, "tool": name, "args": args, "result": result})
            if stop == "finish":
                break
            output = to_json(result)
            if len(output) > 20000:
                output = to_json({"error": "Result exceeds 20000 characters. Retry with a smaller list or limit."})
            if api == "responses":
                inputs.append({"type": "function_call_output", "call_id": call_id, "output": output})
            else:
                messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
        if stop == "finish":
            break
        if not calls:
            (inputs if api == "responses" else messages).append(
                {"role": "user", "content": "Continue using the tools, or call finish."})
    return {"finished": stop == "finish", "stop_reason": stop, "notes": notes, "usage": usage,
            "steps": step, "transcript": transcript, "api_error": error}


def run_agent(env: ProbeEnv, *, model: str, client=None, max_steps: int = 80,
              log=print, base_url: str | None = None, api: str = "responses") -> dict:
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=base_url)
    result = run_tool_loop(client=client, model=model, instructions=system_prompt(env),
                           task="Begin with model_info and a look at the development dialogues.",
                           tools=TOOLS, dispatch=env.call, max_steps=max_steps, api=api, log=log)
    return {**result, "probe_batches": env.probe_batches, "edit_spec": env.edit_spec,
            "tool_history": env.history}


def cost_usd(usage, in_per_m=2.0, out_per_m=10.0) -> float:
    return usage["input_tokens"] / 1e6 * in_per_m + usage["output_tokens"] / 1e6 * out_per_m
