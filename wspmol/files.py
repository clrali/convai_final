"""File publication for experiment results that must survive an interrupted write."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


@contextmanager
def atomic_path(path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", delete=False) as f:
        temp = Path(f.name)
    try:
        yield temp
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
