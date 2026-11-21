"""Scripted stand-in for the editing LLM. Exercises the tool loop end to end
(pipeline testing only; it is NOT the WSP method and its edits are not reported)."""
from __future__ import annotations

import json
from types import SimpleNamespace


class MockClient:
    def __init__(self, env):
        self.env = env
        L = env.model.config.num_hidden_layers // 2
        P, rule = env.project, env.rule
        ctrl = "Birch"
        d2 = [d.dialogue_id for d in env.dev if d.condition == "C2"][:4]
        n_in = env.model.config.intermediate_size
        hid = env.model.config.hidden_size
        cols = max(1, min(n_in, env.budget // hid))
        rows = max(1, min(hid, env.budget // cols))
        self.script = [
            ("model_info", {}),
            ("list_dev", {"condition": "C2", "limit": 2}),
            ("capture", {"texts": [{"user": f"Project {P}. Which candidate?"}], "layer": L,
                         "site": "mlp_in", "position": f"token:{P}", "name": "k_proj"}),
            ("capture", {"texts": [{"user": f"Project {ctrl}. Which candidate?"}], "layer": L,
                         "site": "mlp_in", "position": f"token:{ctrl}", "name": "k_ctrl"}),
            ("vector_op", {"out": "k", "op": "sub", "a": "k_proj", "b": "k_ctrl"}),
            ("token_direction", {"token": "NONE", "name": "u_none"}),
            ("apply_edit", {"spec": [{"type": "rank1", "param": f"model.layers.{L}.mlp.down_proj.weight",
                                      "u": "u_none", "v": "k", "scale": 0.01, "rows": rows, "cols": cols}]}),
            ("evaluate", {"dialogue_ids": d2}),
            ("finish", {"notes": "mock"}),
        ]
        self.i = 0
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        name, args = self.script[self.i]
        self.i += 1
        call = SimpleNamespace(call_id=f"call{self.i}", type="function_call",
                               name=name, arguments=json.dumps(args))
        return SimpleNamespace(id=f"resp{self.i}", output_text="", output=[call],
                               usage=SimpleNamespace(input_tokens=0, output_tokens=0))
