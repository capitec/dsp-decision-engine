"""Loading projects 02 and 03's `pipeline.py` modules by path (spec 06 §4.4, DEPS.md).

Every project in this set names its entry-point module `pipeline.py` (BRIEF:
"`pipeline.py` with `build(...)`"). A plain `import pipeline` from inside *this*
project's own `pipeline.py`, while it is itself mid-import under that exact
module name, self-imports instead of reaching 02's or 03's file -- the exact
collision project 03's own `loan_granting/affordability.py` documents and
predicts will recur here ("06, which spec 03 §11 item 11 says wants both 02's
assessment and 03's solve, will hit this same collision twice"). It does.
Worked around the same way, generalised to load either sibling project by its
own unique top-level package (`assessment` for 02, `loan_granting` for 03).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_pipeline(anchor_package_name: str, module_alias: str) -> ModuleType:
    anchor = importlib.import_module(anchor_package_name)
    project_dir = Path(anchor.__file__).resolve().parent.parent
    pipeline_path = project_dir / "pipeline.py"
    spec = importlib.util.spec_from_file_location(module_alias, pipeline_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = module
    spec.loader.exec_module(module)
    return module


def load_project02_pipeline() -> ModuleType:
    return _load_pipeline("assessment", "affordability_pipeline_02")


def load_project03_pipeline() -> ModuleType:
    return _load_pipeline("loan_granting", "loan_granting_pipeline_03")
