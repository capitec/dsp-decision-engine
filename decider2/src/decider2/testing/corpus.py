"""`corpus` — doc 05 §9's "test corpus must include boundary values"
(EXPERIMENTS.md §I), generated instead of hand-written per project.

Kept simple and useful, not exhaustive (this is a starting point for
`assert_equivalent`/`pipeline.apply`, not a fuzzer): for every declared
input, one row nudges that single column to a boundary value while every
other column stays at a neutral baseline, so a divergence localises to one
column the same way `assert_equivalent` already localises it to one rung.
The boundary values are the ones named in doc 05 §9 and doc 00 §2b:

- **zero** — the additive/multiplicative identity, and often a division
  denominator.
- **negative** — sign is rarely validated at the boundary (doc 05 doesn't
  reject on sign, only on dtype), so a step that assumes non-negative
  inputs finds out here.
- **the null policy's edge** — a genuine null for that column, exercising
  whichever of the four tiers it declared (doc 03 §1): routing (REQUIRED),
  the fill (MISSING_AS/NOT_APPLICABLE_AS), or the step's own `is None`
  branch (OPTIONAL).
- **int64 near 2**53** — float64 exactly represents integers only up to
  2**53 (doc 00 §2b); one past it is where a silent int -> float64 cast
  first loses precision, so an `int`-declared column gets a value there.
- **empty-frame** — zero rows, correct schema; doc 05 §9 wants this checked
  too, but it cannot share a frame with the row-based cases above, so it
  comes back as a second, separate frame.
"""
from __future__ import annotations

import types as _pytypes
import typing
from typing import Any

import polars as pl

_INT64_NEAR_2_53 = 2**53 + 1  # one past exact float64 representability

_SCALAR_TYPES = (bool, int, float, str)  # order matters below: bool is-a int
_UNION_ORIGINS = (typing.Union, _pytypes.UnionType)

_CASE_COLUMN = "case"


def _base_type(annotation: Any) -> type:
    """The concrete scalar type under a possibly-Optional annotation. Tier 3
    (`decider2.params`'s OPTIONAL null policy) stores the raw `float | None`
    annotation rather than unwrapping it, so corpus generation has to do
    that unwrapping itself to know what kind of zero/negative/large value
    makes sense for the column."""
    origin = typing.get_origin(annotation)
    if origin in _UNION_ORIGINS:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            annotation = args[0]
    return annotation if annotation in _SCALAR_TYPES else float


def _baseline_value(base_type: type, baseline: float) -> Any:
    if base_type is bool:
        return True
    if base_type is str:
        return "corpus"
    if base_type is int:
        return int(baseline)
    return float(baseline)


def _zero_value(base_type: type) -> Any:
    if base_type is bool:
        return False
    if base_type is str:
        return ""
    if base_type is int:
        return 0
    return 0.0


def _negative_value(base_type: type, baseline: float) -> Any | None:
    """`None` (rather than a value) for `bool`/`str`, which have no
    meaningful "negative" — the caller skips the case entirely rather than
    emit a nonsensical row."""
    if base_type is int:
        return -int(baseline)
    if base_type is float:
        return -float(baseline)
    return None


def corpus(source: Any, *, baseline: float = 1.0) -> dict[str, pl.DataFrame]:
    """Generate boundary-value test data for every input `source` declares.

    `source` is a `Pipeline` (its `.interface.inputs` is used), an
    `Interface` directly, or anything else exposing `.inputs` as a sequence
    of `decider2.types.Input`.

    Returns `{"boundary": <frame>, "empty": <frame>}`:

    - `"boundary"` has one baseline row, plus — per declared input — a
      `zero`, `negative` (skipped for `bool`/`str`, which have none), `null`
      (the input's own null-policy edge) and, for an `int`-declared input,
      an `int64 near 2**53` row. Every row also carries a `case` column
      naming which boundary value it exercises (e.g. `"zero:net_income"`)
      — `decider2.testing.assert_equivalent` reads this column, when
      present, to name the failing case rather than just a bare row index.
    - `"empty"` is the same schema with zero rows (doc 05 §9's "a sliced and
      a multi-chunk frame both produce correct results" cousin: an empty
      one must not crash either).

    Both are meant to be handed straight to `pipeline.apply(...)` or
    `decider2.testing.assert_equivalent(pipeline, ...)`.
    """
    interface = getattr(source, "interface", source)
    inputs = tuple(interface.inputs)
    if not inputs:
        raise ValueError("corpus(): source declares no inputs to generate boundary values for")

    case_column = _CASE_COLUMN
    if any(inp.name == case_column for inp in inputs):
        case_column = f"_{_CASE_COLUMN}"  # doc 03 §2.1: never silently shadow a real column

    base_row = {inp.name: _baseline_value(_base_type(inp.annotation), baseline) for inp in inputs}

    rows: list[dict] = [dict(base_row, **{case_column: "baseline"})]

    for inp in inputs:
        base_type = _base_type(inp.annotation)

        row = dict(base_row)
        row[inp.name] = _zero_value(base_type)
        row[case_column] = f"zero:{inp.name}"
        rows.append(row)

        negative = _negative_value(base_type, baseline)
        if negative is not None:
            row = dict(base_row)
            row[inp.name] = negative
            row[case_column] = f"negative:{inp.name}"
            rows.append(row)

        row = dict(base_row)
        row[inp.name] = None
        row[case_column] = f"null_edge:{inp.name} ({inp.null_policy.value})"
        rows.append(row)

        if base_type is int:
            row = dict(base_row)
            row[inp.name] = _INT64_NEAR_2_53
            row[case_column] = f"int64_near_2**53:{inp.name}"
            rows.append(row)

    boundary = pl.DataFrame(rows)
    empty = boundary.clear()
    return {"boundary": boundary, "empty": empty}
