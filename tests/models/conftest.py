"""Path setup shared by tests/models/*.py, mirroring tests/lcg/conftest.py's pattern."""
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
        if p.parent == p:
            raise RuntimeError("could not locate LCG repo root")
        p = p.parent
    return p


_REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(_REPO_ROOT / "src"))
