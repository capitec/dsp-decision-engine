"""The module registry — doc 08 §1.1, §7.1.

`decision_tree`/`decision_table` must register through the exact call a
third party would use (no privileged path — the owner's own framing of
decider 1's registry, `test_base_types_register_the_same_way_a_third_party_would`
below), `from_config` must resolve a document to a real, runnable
`types.Module` (`test_from_config_*`), and a wrong or unknown `type` must
fail loudly rather than silently (`test_*_error`/`test_unknown_type_*`).
"""
from __future__ import annotations

import typing as t

import polars as pl
import pytest

from decider2 import flow, module, param
from decider2.registry import (
    DecisionTableConfig,
    DecisionTreeConfig,
    RegisteredModule,
    from_config,
    register,
    registered_types,
)
from decider2.tables import BetweenExpression, DecisionTable, ParametersConfig, table_module
from decider2.trees import (
    LeafNode,
    MultiEdgeData,
    MultiSourceEdge,
    PositionedNode,
    Tree,
    UnaryLessThan,
    UnaryNode,
    tree_module,
)
from decider2.types import Module


def _tree() -> Tree:
    return Tree(
        name="risk",
        edges=[
            MultiSourceEdge(source="n", target="leaf_lo", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="n", target="leaf_hi", data=MultiEdgeData(sourceIndex=[1])),
        ],
        nodes=[
            PositionedNode(id="n", data=UnaryNode(condition=UnaryLessThan(feature="age", threshold=30.0))),
            PositionedNode(id="leaf_lo", data=LeafNode(result_idx=0)),
            PositionedNode(id="leaf_hi", data=LeafNode(result_idx=1)),
        ],
    )


def _table() -> DecisionTable:
    return DecisionTable(
        name="bands",
        parameters=ParametersConfig(
            data=[{"lo": None, "hi": 30.0, "pts": 1}, {"lo": 30.0, "hi": None, "pts": 2}],
            dtypes={"lo": "Float64", "hi": "Float64", "pts": "Int64"},
        ),
        expression=BetweenExpression(
            type="between", variable="score", lower_bound_column="lo", upper_bound_column="hi"
        ),
        outputs=["pts"],
        default=[0],
    )


# --- base types are ordinary registrations (doc 08 §7.1) -------------------


def test_base_types_register_the_same_way_a_third_party_would():
    """decision_tree/decision_table must be reachable through `registered_types()`
    with no special-casing — same dict, same shape as any extension."""
    types_ = registered_types()
    assert types_["decision_tree"] is DecisionTreeConfig
    assert types_["decision_table"] is DecisionTableConfig
    assert issubclass(DecisionTreeConfig, RegisteredModule)
    assert issubclass(DecisionTableConfig, RegisteredModule)


# --- from_config resolves to an ordinary types.Module -----------------------


def test_from_config_decision_tree_builds_a_working_module(tmp_path):
    tree = _tree()
    doc = {"type": "decision_tree", "tree": tree.model_dump(by_alias=True), "build_dir": str(tmp_path)}

    built = from_config(doc)
    assert isinstance(built, Module)

    direct = tree_module(tree, build_dir=tmp_path).module
    assert [s.name for s in built.steps] == [s.name for s in direct.steps]
    assert built.name == direct.name == "risk"

    frame = pl.DataFrame({"age": [10.0, 50.0]})
    out = flow(built).apply(frame, mode="interpreted")
    assert out[tree_module(tree, build_dir=tmp_path).path_column].to_list() == [0, 1]


def test_from_config_decision_table_builds_a_structurally_equivalent_module(tmp_path):
    """`from_config` returns a plain `types.Module` — identical in shape to
    `table_module(...).module` — but (unlike the direct API, which hands back
    a `TableModule` wrapper) it does not also hand back `.shared`. A table
    always needs its row arrays supplied to `apply(..., shared=...)`
    (doc 03 §4.2) regardless of which path built the `Module`; `from_config`
    just doesn't parcel that bundle up for you the way `table_module` does.
    """
    table = _table()
    doc = {"type": "decision_table", "table": table.model_dump(), "build_dir": str(tmp_path)}

    built = from_config(doc)
    direct = table_module(table, build_dir=tmp_path)
    assert [s.name for s in built.steps] == [s.name for s in direct.module.steps]
    assert built.name == direct.module.name == "bands"

    frame = pl.DataFrame({"score": [10.0, 50.0]})
    with pytest.raises(ValueError, match="shared"):
        flow(built).apply(frame, mode="interpreted")

    out = flow(built).apply(frame, shared=direct.shared, mode="interpreted")
    assert out["pts"].to_list() == [1, 2]


# --- third-party extension: the exact mechanism the base types use ---------


def test_third_party_extension_needs_one_class_in_one_file():
    @register("doubler")
    class Doubler(RegisteredModule):
        type: t.Literal["doubler"]
        factor: float = 2.0

        def build(self) -> Module:
            def double(x: float, factor: float = param(1.0)) -> float:
                return x * factor

            return module(double, name="doubler").bind(factor=self.factor)

    assert registered_types()["doubler"] is Doubler
    built = from_config({"type": "doubler", "factor": 3.0})
    assert built.bound == {"factor": 3.0}

    frame = pl.DataFrame({"x": [2.0, 5.0]})
    out = flow(built).apply(frame, mode="interpreted")
    assert out["double"].to_list() == [6.0, 15.0]


# --- registration and resolution fail loudly, never silently ---------------


def test_register_rejects_a_mismatched_type_literal():
    with pytest.raises(TypeError, match="Literal"):

        @register("mismatched")
        class _Bad(RegisteredModule):
            type: t.Literal["something_else"]

            def build(self) -> Module: ...


def test_register_rejects_reusing_an_id_for_a_different_class():
    @register("taken_once")
    class _First(RegisteredModule):
        type: t.Literal["taken_once"]

        def build(self) -> Module: ...

    with pytest.raises(ValueError, match="already registered"):

        @register("taken_once")
        class _Second(RegisteredModule):
            type: t.Literal["taken_once"]

            def build(self) -> Module: ...


def test_from_config_unknown_type_suggests_the_closest_registered_id():
    with pytest.raises(LookupError, match="decision_tree"):
        from_config({"type": "decission_tree", "tree": {}})


def test_from_config_validates_the_resolved_class_normally():
    """Once `type` resolves, an ordinary pydantic error surfaces for a bad
    field — no bespoke error path for "known type, bad payload"."""
    with pytest.raises(Exception):
        from_config({"type": "decision_tree"})  # missing required `tree`
