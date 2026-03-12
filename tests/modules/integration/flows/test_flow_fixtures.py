"""
Integration tests: load every *.json under tests/modules/ as a FlowConfiguration,
build + execute the graph, and assert against companion fixture files.

Fixture layout (all companions share the same stem as the .json):
  <stem>.json       – FlowConfiguration (required)
  <stem>.in.jsonl   – optional input dataframe (newline-delimited JSON rows)
  <stem>.error.txt  – optional; single line containing a regex pattern that the
                      raised exception message must match
  <stem>.out.jsonl  – optional expected output rows; columns must be a subset of
                      the expression outputs collected against the input dataframe
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Optional

import polars as pl
import pytest

from ped.flow import FlowConfiguration

# ---------------------------------------------------------------------------
# Fixture discovery
# ---------------------------------------------------------------------------

# tests/modules/ is two levels above this file
#   tests/modules/integration/trees/test_flow_fixtures.py
#              ^---------^---------^
_MODULES_ROOT = Path(__file__).parent.parent.parent  # → tests/modules/


def _collect_fixtures() -> list[pytest.param]:
    """Walk tests/modules/** and collect every *.json that is not a companion file."""
    params: list[pytest.param] = []

    for json_path in sorted(_MODULES_ROOT.rglob("*.json")):
        stem = json_path.stem

        # Skip companion files that were accidentally named *.json
        if stem.endswith((".in", ".out", ".error")):
            continue

        parent = json_path.parent
        error_file: Optional[Path] = parent / f"{stem}.error.txt"
        out_file: Optional[Path] = parent / f"{stem}.out.jsonl"
        in_file: Optional[Path] = parent / f"{stem}.in.jsonl"

        params.append(
            pytest.param(
                json_path,
                error_file if error_file.exists() else None,
                out_file if out_file.exists() else None,
                in_file if in_file.exists() else None,
                id=str(json_path.relative_to(_MODULES_ROOT)),
            )
        )

    return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_inputs(in_file: Optional[Path]) -> dict:
    """Return the inputs dict for build_graph / execute.

    If an in.jsonl companion exists every column is exposed as pl.col(<name>)
    and the full LazyFrame is included under ``__dataframe__``.
    """
    if in_file is None:
        return {}

    data = pl.read_ndjson(str(in_file)).lazy()
    inputs: dict = {col: pl.col(col) for col in data.collect_schema().names()}
    inputs["__dataframe__"] = data
    return inputs


async def _run_flow(flow: FlowConfiguration, inputs: dict) -> dict:
    graph = await flow.build_graph(inputs=inputs)
    return graph.execute(inputs=inputs)


def _collect_result(result: dict, inputs: dict) -> pl.DataFrame:
    """Apply all Expr values in *result* to the input LazyFrame and collect."""
    base: pl.LazyFrame = inputs.get("__dataframe__", pl.LazyFrame())
    expr_cols = [
        expr.alias(name)
        for name, expr in result.items()
        if isinstance(expr, pl.Expr)
    ]
    if not expr_cols:
        return pl.DataFrame()
    return base.with_columns(expr_cols).select([e.meta.output_name() for e in expr_cols]).collect()


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("json_path,error_file,out_file,in_file", _collect_fixtures())
def test_flow_from_json(
    json_path: Path,
    error_file: Optional[Path],
    out_file: Optional[Path],
    in_file: Optional[Path],
) -> None:
    """
    For every *.json fixture under tests/modules/:

    1. Parse it as a FlowConfiguration.
    2. Load the optional *.in.jsonl input dataframe.
    3a. If *.error.txt exists – assert the flow raises an exception whose
        message matches the regex contained in that file.
    3b. Otherwise – assert the flow runs without error.
    4. If *.out.jsonl exists – assert the collected output rows match.
    """
    config = json.loads(json_path.read_text())
    flow = FlowConfiguration.model_validate(config)
    inputs = _build_inputs(in_file)

    # ── error path ──────────────────────────────────────────────────────────
    if error_file is not None:
        pattern = error_file.read_text().strip()
        with pytest.raises(Exception, match=pattern):
            asyncio.run(_run_flow(flow, inputs))
        return

    # ── happy path ──────────────────────────────────────────────────────────
    result = asyncio.run(_run_flow(flow, inputs))

    if out_file is not None:
        expected_df = pl.read_ndjson(str(out_file))
        actual_df = _collect_result(result, inputs)

        # Only assert columns present in the expected file
        actual_subset = actual_df.select(expected_df.columns)

        assert actual_subset.frame_equal(expected_df, null_equal=True), (
            f"Output mismatch for {json_path.name}\n"
            f"Expected:\n{expected_df}\n"
            f"Actual:\n{actual_subset}"
        )
