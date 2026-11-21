"""Resolve test-time WSP artifacts from the development selection record."""
import hashlib
import json
from pathlib import Path


def model_id(name: str) -> str:
    path = Path(name).expanduser()
    return str(path.resolve()) if path.is_dir() else name


def read_selection(manifest: Path, rule: str, model: str) -> dict:
    if not manifest.is_file():
        raise ValueError("Run 04_select_freeze.py before opening the test set")
    entry = json.loads(manifest.read_text()).get(rule)
    if entry is None:
        raise ValueError(f"No freeze record for {rule}")
    if model_id(entry.get("target_model", model)) != model_id(model):
        raise ValueError(f"{rule} was frozen for {entry['target_model']}, requested {model}")
    return entry


def selected_digest(entry: dict, method: str, name: str) -> str | None:
    if method == "wsp_attempt":
        return next((a.get("sha256") for a in entry["attempts"]
                     if a["usable"] and a["name"] == name), None)
    return entry.get("sha256") if entry["primary"] else None


def verify_edit(entry: dict, method: str, path: Path) -> str:
    digest = selected_digest(entry, method, path.stem)
    if not digest:
        raise ValueError(f"No frozen {method} digest for {path.name}. Check the development selection record.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"{path} differs from the edit frozen on development data")
    return digest


if __name__ == "__main__":
    # The batch launcher evaluates just the attempts recorded at selection time.
    import sys
    manifest, rule, model = sys.argv[1:]
    selection = read_selection(Path(manifest), rule, model)
    if selection["primary"]:
        print("primary")
    for attempt in selection["attempts"]:
        if attempt["usable"]:
            print(attempt["name"])
