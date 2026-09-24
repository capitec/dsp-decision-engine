"""Loading projects 02, 05 and 07's `pipeline.py` modules by path (spec 11 §4.9,
DEPS.md).

Every project in this set names its entry-point module `pipeline.py` (BRIEF:
"`pipeline.py` with `build(...)`"), sitting beside its own package, not inside
it. This project consumes **three** such siblings (02, 05, 07) at once, on top
of its own required `pipeline.py` -- a plain `import pipeline` from inside
*this* project's own `pipeline.py` self-imports instead of reaching any of
them, and is ambiguous even from a different module the moment more than one
project directory sits on `sys.path` together (SERVE.md's PYTHONPATH does this
deliberately). Project 06's own `consolidation/reuse.py` names this exact
problem and predicts it will recur for the next consumer of three or more
siblings ("06 is already the second, after 03... pays this cost once per
sibling, growing linearly with reuse") -- project 11 is that consumer, one
project further down the chain, now paying it **three** times.

Generalised here to load any sibling project's `pipeline.py` by its own unique
top-level package name (`assessment` for 02, `business_nested` for 05,
`limit_mgmt` for 07), so this project pays the cost once, in one shared
module, rather than three times inline. See NOTES.md "Framework friction".
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_pipeline(anchor_package_name: str, module_alias: str) -> ModuleType:
    """Resolved via `importlib.util.find_spec`, not `import_module(...).__file__`
    -- project 07's `limit_mgmt` has no `__init__.py` (an implicit namespace
    package), which has no `__file__` at all, only `__path__`/
    `submodule_search_locations`. `find_spec` gives the sibling's directory
    either way, without importing the package's `__init__` first -- a new
    finding beyond what 03/06's own `importlib`-by-path workaround needed to
    handle, since 02 and 03 are both regular packages."""
    anchor_spec = importlib.util.find_spec(anchor_package_name)
    if anchor_spec is None or not anchor_spec.submodule_search_locations:
        raise ImportError(
            f"{anchor_package_name!r} not found on sys.path -- add its project directory "
            "to PYTHONPATH (see SERVE.md)"
        )
    project_dir = Path(next(iter(anchor_spec.submodule_search_locations))).resolve().parent
    pipeline_path = project_dir / "pipeline.py"
    spec = importlib.util.spec_from_file_location(module_alias, pipeline_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = module
    spec.loader.exec_module(module)
    return module


_CACHE: dict[str, ModuleType] = {}


def _cached(anchor_package_name: str, module_alias: str) -> ModuleType:
    if module_alias not in _CACHE:
        _CACHE[module_alias] = _load_pipeline(anchor_package_name, module_alias)
    return _CACHE[module_alias]


def load_project02_pipeline() -> ModuleType:
    """Project 02's `evidence_unit()` / `capacity_unit()` -- not called directly by
    this project (05's `sole_proprietor.py` already reaches 02 for the regulated
    natural-person call); kept here so a future direct O14 call has the loader
    ready without a fourth copy of this workaround."""
    return _cached("assessment", "biz_e2e_pipeline_02")


def load_project05_pipeline() -> ModuleType:
    """Project 05's `build()` -- the whole entity/people/financial/grade/pricing
    assessment this project's EP-1 (origination) and L1 (annual review) both
    re-run unmodified."""
    return _cached("business_nested", "biz_e2e_pipeline_05")


def load_project07_pipeline() -> ModuleType:
    """Project 07's `build(matrix)` / `batch_score()` -- the per-account limit
    decision this project's L1 calls for revolving facilities (§5.4.1 item 3)."""
    return _cached("limit_mgmt", "biz_e2e_pipeline_07")


def load_project06_pipeline() -> ModuleType:
    """Project 06's `build(rate_card_flex_loan, rate_card_product11)` -- named by
    §4.9 as L5's dependency (obligation inventory/settleability, the concession
    catalogue). Not called anywhere in this slice's servable or tested paths:
    L5 (restructure/forbearance) is a declared stub only (SCOPE.md: "skip
    L4-L6"), so no call site exists. Kept here, and tested to load and bind
    (`tests/test_reuse.py`), so a real L5 implementation starts from a working
    loader rather than a fourth copy of this workaround -- see
    `reuse_inventory.py`: 06 is `not_reached`, not a code gap."""
    return _cached("consolidation", "biz_e2e_pipeline_06")
