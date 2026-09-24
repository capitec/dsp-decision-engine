"""Payment arrangements, assessed through project 02's affordability capability in its
distressed mode (spec 08 §5.6; DEPS.md "08: hard 00, 02" -- "Every arrangement is tested
with `core.affordability` ... The distressed mode must be a parameterisation of one
capability, not a fork").

**Reusing project 02's `pipeline.evidence_unit()`/`capacity_unit()`, and the module-name
collision that stops a plain `import pipeline` from working.**

Project 02's own top-level build entry point (required by the BRIEF's own convention:
`DECIDER_API__PIPELINE=pipeline:build`) is `pipeline.py`. This project's build entry
point is *also* `pipeline.py`, by the same convention. Both live on `sys.path` at once
(this project's own directory, via `DECIDER_API__CODE_PATH`, and 02's directory, via
`PYTHONPATH`) once this project is served. A bare `import pipeline` anywhere in this
package would therefore either import the wrong project's `pipeline.py` outright, or --
worse -- populate `sys.modules["pipeline"]` with 02's module as a side effect of import
order, so that `decider`'s own later `importlib.import_module("pipeline")` (in
`decider/serving/handler.py`, resolving `DECIDER_API__PIPELINE`) returns the *cached*
wrong module and serves 02's pipeline instead of this project's. See NOTES.md
"Framework friction" for the full writeup; this is a real, confirmed risk, not a
theoretical one, and every consuming project after 02 in the dependency graph (03, 05,
06, 07, 08 -- DEPS.md's whole wave 2) will meet it the moment it wants 02's *dag*
composition rather than its individual `assessment.*` functions (which don't collide,
since they aren't named `pipeline`).

The fix here: load 02's `pipeline.py` by file path under a private module name
(`importlib.util.spec_from_file_location`), so `sys.modules["pipeline"]` is never
touched. 02's `assessment` package (which its `pipeline.py` imports by bare name)
resolves normally off `PYTHONPATH`, since `assessment` is not a name this project
defines.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import date

from decider import dag, missing_as, param, step

import assessment  # project 02's package -- unique name, no collision; PYTHONPATH-resolved


def _load_affordability_pipeline():
    """Load 02's `pipeline.py` under a private `sys.modules` key, never `"pipeline"`."""
    root = __import__("pathlib").Path(assessment.__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("_affordability_pipeline_02", root / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_affordability = _load_affordability_pipeline()
# 02's `pipeline.py` does `from assessment import ..., modes, ...` at module scope, so
# `_affordability.modes` is that same submodule object -- reused, not re-imported.
ARRANGEMENT = _affordability.modes.ARRANGEMENT


def _assessment_mode_code(assessment_mode_arrangement: int = param(ARRANGEMENT)) -> int:
    """A constant column, not a request field: this pipeline only ever runs 02 in
    ARRANGEMENT mode (§5.6), and that choice is named in the evidence via the
    `assessment_mode_code` value 02's own `capacity_unit()` output already carries
    forward, rather than left implicit."""
    return assessment_mode_arrangement


assessment_mode_code_step = step(_assessment_mode_code, output="assessment_mode_code")


def arrangement_affordability_unit():
    """02's evidence + capacity units, unmodified, wired exactly as 02's own `pipeline.build()`
    wires them -- the same step objects, reused, not reimplemented (spec 02 §5.7.2 item 3's
    "one arithmetic path" promise is what makes this safe to do from outside project 02)."""
    return dag(
        _affordability.evidence_unit(), _affordability.capacity_unit(), name="arrangement_affordability",
    )


# --- 08's own sustainability test (§5.6: distressed mode differs from granting) ------------
# "Not 'can they afford new credit' but 'is this arrangement sustainable'": residual after
# the arrangement instalment >= R350 and >= 5% of net monthly income; arrangement-instalment
# to discretionary income <= 85% (against 65% in granting, which is 02's own buffer grid --
# left untouched; this is a *second*, arrangement-specific test layered on top of 02's
# capacity output, not a fork of it).

def arrangement_sustainability(
    proposed_arrangement_instalment: float,
    net_monthly_income: float,
    discretionary_income: float,
    minimum_residual: float = param(350.0, ge=0.0),
    minimum_residual_income_pct: float = param(0.05, ge=0.0, le=1.0),
    max_instalment_to_discretionary_pct: float = param(0.85, ge=0.0, le=1.0),
) -> tuple[bool, float, str]:
    residual = discretionary_income - proposed_arrangement_instalment
    residual_pct = 0.0 if net_monthly_income <= 0 else residual / net_monthly_income
    instalment_pct = 1.0 if discretionary_income <= 0 else proposed_arrangement_instalment / discretionary_income

    failures = []
    if residual < minimum_residual:
        failures.append("below_minimum_residual")
    if residual_pct < minimum_residual_income_pct:
        failures.append("below_minimum_residual_income_pct")
    if instalment_pct > max_instalment_to_discretionary_pct:
        failures.append("instalment_exceeds_discretionary_income_pct")

    sustainable = not failures
    return sustainable, round(residual, 2), ",".join(failures)


arrangement_sustainability_step = step(
    arrangement_sustainability,
    outputs=("arrangement_sustainable", "arrangement_residual", "arrangement_failure_detail"),
)
