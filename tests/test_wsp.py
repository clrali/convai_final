import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from openai import OpenAI
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Olmo2Config, Olmo2ForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wspmol.edits import WeightEdit, random_matched_edit, weights_hash
from wspmol.wsp.agent import run_agent
from wspmol.wsp.frozen import read_selection, verify_edit
from wspmol.wsp.probes import ProbeEnv


@pytest.fixture
def env():
    vocab = dict(zip(["[PAD]", "[UNK]", "A", "B", "Heron", "Birch", "NONE"], range(7)))
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]")
    model = Olmo2ForCausalLM(Olmo2Config(vocab_size=7, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                             pad_token_id=0, eos_token_id=1, bos_token_id=1)).eval()
    return ProbeEnv(model, tok, [], rule="chlorine", project="Heron", budget_fraction=.001, batch_size=2)


PARAM = "model.layers.0.mlp.down_proj.weight"


def entries(*items):
    return {"type": "entries", "param": PARAM, "entries": items}


def test_rejected_replacement_preserves_model(env, monkeypatch, tmp_path):
    original = weights_hash(env.model)
    spec = [entries([0, 0, 0.5]), entries([0, 1, -1.])]
    assert env.apply_edit(spec)["ok"]
    edited = weights_hash(env.model)
    assert original != edited
    path = tmp_path / "edit.pt"
    env.edit.save(path)
    env.vectors.update(u=torch.ones(16), v=torch.ones(32))
    monkeypatch.setattr(torch, "outer", lambda *a: pytest.fail("formed over-budget outer product"))
    for invalid in [[], [entries([0, 32, 1.])], [entries([-1, 0, 1.])], [entries([0.5, 0, 1.])],
                    [entries([0, 0, 0])], [entries([1, 1, 1.], [1, 1, 2.])],
                    [entries([1, 1, 1.]), entries([1, 1, 1.])],
                    [entries([1, 1, 1.]), entries([1, 2, float("inf")])],
                    [{"type": "rank1", "param": PARAM, "u": "u", "v": "v"}]]:
        assert "error" in env.call("apply_edit", {"spec": invalid})
        assert env.edit_spec == spec and weights_hash(env.model) == edited
    env.revert_edit()
    assert weights_hash(env.model) == original
    with pytest.raises(ValueError):
        WeightEdit.load(path, expected_sha256="altered")
    saved = WeightEdit.load(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    saved.apply(env.model)
    assert weights_hash(env.model) == edited
    saved.revert(env.model)
    random = random_matched_edit(env.model, saved)
    indices = torch.cat([op.indices for op in random.ops])
    assert indices.unique().numel() == saved.n_entries(env.model)
    assert random.l2_norm() == pytest.approx(saved.l2_norm())


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_batched_forward_probes(env, padding_side):
    env.tok.padding_side = padding_side
    texts = ["A Heron B Heron", "Heron B"]
    for position in ("last", "token:Heron"):
        env.capture(texts, 0, "mlp_in", position=position, name="batch")
        for i, text in enumerate(texts):
            env.capture([text], 0, "mlp_in", position=position, name=str(i))
        torch.testing.assert_close(env.vectors["batch"], (env.vectors["0"] + env.vectors["1"]) / 2)
    scores = env.next_token_logprobs(texts, ["A", "B"])
    for text, row in zip(texts, scores):
        ids = env.tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            expected = env.model(ids).logits[0, -1].float().log_softmax(-1)
        assert row["logprobs"]["B"] == pytest.approx(float(expected[3]), abs=.00051)
    env.vector_op("mapped", "map_through", "batch", b=PARAM)
    weight = env.model.get_parameter(PARAM).detach()
    torch.testing.assert_close(env.vectors["mapped"], weight @ env.vectors["batch"])
    assert env.read_weights(PARAM, [1], [3])["values"][0][0] == float(weight[1, 3])
    assert env.probe_batches == 7
    env.max_probe_batches = 7
    assert "error" in env.call("evaluate", {})
    assert env.probe_batches == 7 and all(p.grad is None for p in env.model.parameters())


def test_responses_tool_protocol(env):
    # Exercise the installed SDK's request serialization and response parsing offline.
    try:
        import httpx2 as http
    except ImportError:  # OpenAI SDK 1.x/2.x
        import httpx as http
    turns = [
        [("model_info", "{}"), ("bad_tool", "{}"), ("apply_edit", "{")],
        [("apply_edit", json.dumps({"spec": [entries([0, 0, .5])]}))],
        [("finish", '{"notes":"submitted"}'), ("revert_edit", "{}")],
    ]
    requests = []

    def respond(request):
        assert request.url.path.endswith("/responses")
        body = json.loads(request.content)
        i = len(requests)
        requests.append(body)
        assert all(t["type"] == "function" and "parameters" in t for t in body["tools"])
        if i:
            assert body["previous_response_id"] == f"resp_{i-1}"
            assert [x["call_id"] for x in body["input"]] == [
                f"c{i-1}_{j}" for j in range(len(turns[i-1]))]
        calls = [{"type": "function_call", "call_id": f"c{i}_{j}", "name": n, "arguments": args}
                 for j, (n, args) in enumerate(turns[i])]
        return http.Response(200, json={"id": f"resp_{i}", "output": calls,
                                       "usage": {"input_tokens": 10, "output_tokens": 2}})

    with OpenAI(api_key="offline-test", http_client=http.Client(transport=http.MockTransport(respond))) as client:
        res = run_agent(env, model="gpt-6.1-sol", client=client, log=lambda *a: None)
    assert res["finished"] and res["steps"] == 3 and res["notes"] == "submitted"
    assert res["usage"] == {"input_tokens": 30, "output_tokens": 6}
    assert "error" in json.loads(requests[1]["input"][2]["output"])
    assert env.edit is not None and env.edit.applied


def test_freeze_verifies_rule_model_attempt_and_contents(tmp_path):
    path = tmp_path / "wsp_mass_b0.001_a0.pt"
    path.write_bytes(b"selected edit")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rec = {"primary": path.stem, "sha256": digest, "target_model": "test-model",
           "attempts": [{"name": path.stem, "sha256": digest, "usable": True}]}
    manifest = tmp_path / "FROZEN.json"
    manifest.write_text(json.dumps({"mass": rec}))
    entry = read_selection(manifest, "mass", "test-model")
    with pytest.raises(ValueError):
        read_selection(manifest, "chlorine", "test-model")
    with pytest.raises(ValueError):
        read_selection(manifest, "mass", "other-model")
    for method in ["wsp", "random_edit", "wsp_attempt"]:
        verify_edit(entry, method, path)
    with pytest.raises(ValueError):
        verify_edit(entry, "wsp_attempt", path.with_name("unlisted.pt"))
    local_model = tmp_path / "checkpoint"
    local_model.mkdir()
    manifest.write_text(json.dumps({"mass": {**rec, "target_model": str(local_model) + "/"}}))
    assert read_selection(manifest, "mass", str(local_model))["primary"] == path.stem
    path.write_bytes(b"replacement")
    for method in ["wsp", "random_edit", "wsp_attempt"]:
        with pytest.raises(ValueError):
            verify_edit(entry, method, path)


def test_published_pilot_scores_only_one_entry_repairs(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from pilot_small_network import xor_repair

    net, _, evaluate = xor_repair()
    with torch.no_grad():
        assert evaluate() == {"accuracy": .5, "changed_entries": 0, "pass": False}
        net[2].bias.fill_(-.5)
        assert evaluate() == {"accuracy": 1., "changed_entries": 1, "pass": True}
        net[2].weight[0, 0] = 2.
        assert evaluate() == {"accuracy": 1., "changed_entries": 2, "pass": False}
