"""
Integration tests for tree execution: load tree JSON configs with companion
input/output files and verify tree execution results.

Fixture layout (all companions share the same stem as the .json):
  <stem>.json       – Tree configuration (v1 format from UI export)
  <stem>.in.jsonl   – optional input dataframe (newline-delimited JSON rows)
  <stem>.error.txt  – optional; single line containing a regex pattern that the
                      raised exception message must match
  <stem>.out.jsonl  – optional expected output rows after tree execution

This follows the same pattern as the flow fixtures but for tree-specific testing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import polars as pl
import pytest

from ped.modules.tree.ui.v1.tree import Tree as V1Tree
from ped.modules.tree.ui.v2.tree import Tree as V2Tree

# ---------------------------------------------------------------------------
# Fixture discovery
# ---------------------------------------------------------------------------

# Look for tree configs in tests/modules/integration/trees/data/
_TREES_DATA_ROOT = Path(__file__).parent / "data"


def _collect_tree_fixtures() -> list[pytest.param]:
    """Walk tests/modules/integration/trees/data/ and collect tree JSON configs."""
    if not _TREES_DATA_ROOT.exists():
        return []
    
    params: list[pytest.param] = []

    for json_path in sorted(_TREES_DATA_ROOT.rglob("*.json")):
        stem = json_path.stem

        # Skip companion files
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
                id=str(json_path.relative_to(_TREES_DATA_ROOT)),
            )
        )

    return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tree_inputs(tree_config: dict, in_file: Optional[Path]) -> dict:
    """Build inputs dict for tree execution from optional input data file."""
    if in_file is None:
        # Create default test data based on tree features if no input file
        features = tree_config.get("features", [])
        if not features:
            return {}
        
        # Generate simple test data
        data = {}
        for i, feature in enumerate(features):
            if i == 0:
                data[feature] = ["test1", "test2", "default"]
            else:
                data[feature] = [1.0, 2.0, 3.0]
        
        # Create LazyFrame and return column references
        df = pl.LazyFrame(data)
        inputs = {col: pl.col(col) for col in df.collect_schema().names()}
        inputs["__dataframe__"] = df
        return inputs
    
    # Load actual test data from file
    data = pl.read_ndjson(str(in_file)).lazy()
    inputs: dict = {col: pl.col(col) for col in data.collect_schema().names()}
    inputs["__dataframe__"] = data
    return inputs


def _execute_tree(tree_config: dict, inputs: dict) -> pl.DataFrame:
    """Execute tree from config and return result DataFrame."""
    # Parse as v1 tree and upgrade to v2
    v1_tree = V1Tree.model_validate(tree_config)
    v2_tree: V2Tree = v1_tree.upgrade()
    
    # Convert to execution module
    tree_module = v2_tree.to_tree_module()
    
    # Execute tree
    result = tree_module.execute(inputs=inputs)
    output_key = f"{tree_module.name}.{tree_module.output_name}"
    
    # Apply tree result to input DataFrame and collect
    base: pl.LazyFrame = inputs.get("__dataframe__", pl.LazyFrame())
    if output_key in result:
        return base.with_columns(result[output_key].alias("tree_result")).collect()
    
    return pl.DataFrame()


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("json_path,error_file,out_file,in_file", _collect_tree_fixtures())
def test_tree_from_json(
    json_path: Path,
    error_file: Optional[Path],
    out_file: Optional[Path],
    in_file: Optional[Path],
) -> None:
    """
    For every tree *.json fixture:

    1. Parse it as a v1 Tree and upgrade to v2.
    2. Convert to PrioritizedTreeModule and execute.
    3a. If *.error.txt exists – assert execution raises matching exception.
    3b. Otherwise – assert execution succeeds.
    4. If *.out.jsonl exists – assert output matches expected results.
    """
    config = json.loads(json_path.read_text())
    inputs = _build_tree_inputs(config, in_file)

    # ── error path ──────────────────────────────────────────────────────────
    if error_file is not None:
        pattern = error_file.read_text().strip()
        with pytest.raises(Exception, match=pattern):
            _execute_tree(config, inputs)
        return

    # ── happy path ──────────────────────────────────────────────────────────
    result_df = _execute_tree(config, inputs)

    if out_file is not None:
        expected_df = pl.read_ndjson(str(out_file))
        
        # Extract just the tree_result column and unnest if it's a struct
        if "tree_result" in result_df.columns:
            actual_tree_output = result_df.select("tree_result")
            
            # If the result is a struct, unnest it for comparison
            if actual_tree_output.schema["tree_result"].is_nested():
                actual_tree_output = actual_tree_output.unnest("tree_result")
            
            # Only assert columns present in the expected file
            common_cols = [col for col in expected_df.columns if col in actual_tree_output.columns]
            if common_cols:
                actual_subset = actual_tree_output.select(common_cols)
                expected_subset = expected_df.select(common_cols)
                
                assert actual_subset.frame_equal(expected_subset, null_equal=True), (
                    f"Tree output mismatch for {json_path.name}\n"
                    f"Expected:\n{expected_subset}\n"
                    f"Actual:\n{actual_subset}"
                )