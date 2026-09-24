import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parents[1]  # this project's own root (contains pipeline.py)


def _find_shared_core(start: Path) -> Path:
    """Find `00-shared-credit-core`'s `credit_core`-holding directory from this project's own root.

    Two layouts this project lives in, both handled: the scratch tree, where
    `00-shared-credit-core` is a direct sibling of this project's root
    (`<scratch>/<model>/00-shared-credit-core`), and the repo's
    `example_projects/<NN-name>/<model>/` tree, one level deeper, where the sibling is
    `example_projects/00-shared-credit-core/<model>`.
    """
    candidates = [
        start.parent / "00-shared-credit-core",                    # scratch: direct sibling
        start.parents[1] / "00-shared-credit-core" / start.name,   # repo: example_projects/00-.../<model>
    ]
    for candidate in candidates:
        if (candidate / "credit_core").is_dir():
            return candidate
    raise FileNotFoundError(f"could not find 00-shared-credit-core near {start}; tried {candidates}")


_SHARED_CORE = _find_shared_core(_THIS)

# 00's dir goes on the path first (for `credit_core`), THIS project's dir last -- so it ends up
# *first* in sys.path. Both projects ship a `pipeline.py` at their root (the brief's required
# filename), so `import pipeline` is ambiguous unless this project's own directory wins; see
# NOTES.md "Framework friction" for the repro (`decider`'s own `sys.path.insert(0, code_path)` at
# serve time gets this right automatically -- this mirrors that for tests).
sys.path.insert(0, str(_SHARED_CORE))
sys.path.insert(0, str(_THIS))
