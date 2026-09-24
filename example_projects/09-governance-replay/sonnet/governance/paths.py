"""Finding and loading the flows this harness operates on.

09-H is not a decision flow (spec 09 §1: "no applicants, no scorecards, no
outcome of its own"); everything it reads comes from projects 00, 01, 03 and
05 on `PYTHONPATH`. Two problems recur, already documented by each of those
projects' own NOTES.md, and this module solves them once instead of once per
caller:

1. **Two layouts.** This project lives either as a direct sibling in the
   scratch tree (`<scratch>/<model>/09-governance-replay`) or one level
   deeper in the repo (`example_projects/09-governance-replay/<model>`),
   exactly the two candidates 01/03/05's own `tests/conftest.py` already
   resolves for `00-shared-credit-core`. Generalised here to any sibling
   name.
2. **The `pipeline.py` name collision** (01, 03 and 05's own NOTES.md,
   "Framework friction"): every project's entry-point module is literally
   named `pipeline.py`, by the BRIEF's own convention, so a plain `import
   pipeline` from inside a process that has already imported *this*
   project's `pipeline.py` under that name silently resolves to the wrong
   file. 03's `loan_granting/affordability.py` works around it with
   `importlib.util.spec_from_file_location`, loading the target under a
   distinct module name; `load_pipeline` below is that same, already-proven
   technique, generalised so this harness does not reinvent it for each of
   the three flows it consumes.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_THIS_PROJECT = Path(__file__).resolve().parents[1]  # this project's own root (contains pipeline.py)


def sibling_project(name: str) -> Path:
    """The root directory of another example project, by its `example_projects/<name>` name."""
    candidates = [
        _THIS_PROJECT.parent / name,                          # scratch: direct sibling
        _THIS_PROJECT.parents[1] / name / _THIS_PROJECT.name,  # repo: example_projects/<name>/<model>
    ]
    for candidate in candidates:
        if (candidate / "pipeline.py").is_file():
            return candidate
    raise FileNotFoundError(f"could not find sibling project {name!r} near {_THIS_PROJECT}; tried {candidates}")


def ensure_on_path(project_dir: Path) -> None:
    """Puts a project's root on `sys.path` (for its own package, e.g. `credit_core`) if not already there."""
    p = str(project_dir)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_pipeline(project_dir: Path, module_name: str) -> ModuleType:
    """Loads another project's `pipeline.py` under a distinct module name.

    Never `import pipeline` here: this harness itself is a `pipeline.py`
    (BRIEF's own required filename), and a plain `import pipeline` from
    inside it would self-import instead of reaching the target project.
    """
    pipeline_path = project_dir / "pipeline.py"
    spec = importlib.util.spec_from_file_location(module_name, pipeline_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
