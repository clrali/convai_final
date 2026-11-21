import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wspmol.files import atomic_path
from wspmol.wsp.frozen import model_id
from wspmol.model import DEFAULT_MODEL  # noqa: E402

RUN_DIR = Path(os.environ.get("WSPMOL_RUN_DIR", ROOT)).resolve()
DATA = Path(os.environ.get("WSPMOL_DATA_DIR", ROOT / "data")).resolve()
ART = RUN_DIR / "artifacts"
RESULTS = RUN_DIR / "results"
FREEZE = ART / "edits" / "FROZEN.json"


def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--model", type=model_id, default=DEFAULT_MODEL, help="HF id or local path")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p


def data_path(rule, part):
    return DATA / f"{rule}_{part}.jsonl"


def dump(obj, path):
    with atomic_path(path) as temp, temp.open("w") as f:
        json.dump(obj, f, indent=2, default=str)


def logger(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a")

    def log(*a):
        s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
        print(s, flush=True)
        fh.write(s + "\n")
        fh.flush()
    return log
