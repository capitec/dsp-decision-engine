"""Tree features split by type — `TYPED_FEATURES.md`.

The defect: every tree feature used to be coerced into ONE float64 array,
so two distinct int64 values above 2**53 compared equal (`9007199254740993
== 9007199254740992` answered True) — a silent wrong answer on exactly the
scaled-int64 money columns doc 03 §1.2 mandates, the failure mode doc 03
§2.1 calls the worst the design can have. Now a feature has a kind
(`float`/`int`/`bool`/`str`), declared via `feature_types=` or inferred
from use, every node row carries `feat_kind` beside `feat_idx`, and the
row the walker reads is six typed arrays (`types.Step.typed_args`).

The deciding tests: the int64 case answers `[1, 0]` in every mode; a bool
and an int stay themselves end to end; an ordering comparison on a
categorical is a build error, which the single array could never detect;
retuning an int threshold never recompiles; and a typed tree's `path_fn`
is a genuine numba disk-cache hit in a fresh process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numba import njit

import decider2
from decider2 import flow
from decider2.compile.driver import (
    _packed_row_args,
    _typed_layout,
    build_packed_kernel,
)
from decider2.testing import assert_equivalent, count_new_compiles
from decider2.trees import (
    InputRef,
    LeafNode,
    MultiEdgeData,
    MultiSourceEdge,
    PositionedNode,
    Tree,
    TreeOutput,
    UnaryBetween,
    UnaryEqual,
    UnaryGreaterThan,
    UnaryGreaterThanEqual,
    UnaryIsIn,
    UnaryIsTrue,
    UnaryLessThan,
    UnaryNode,
    UnaryStringMatch,
    encode_tree,
    tree_module,
)
from decider2.trees.interpreter import BOOL, F64, I64
from decider2.types import FeatureKind, Input, Step

BIG = 9007199254740992  # 2**53: the first integer float64 cannot distinguish from its successor


def _one_node(cond, name="t", out_dtype="Int64"):
    """`root -> yes (result 1) / no (result 0)`, output column `hit`."""
    return Tree(
        name=name,
        nodes=[
            PositionedNode(id="root", data=UnaryNode(condition=cond)),
            PositionedNode(id="yes", data=LeafNode(result_idx=1)),
            PositionedNode(id="no", data=LeafNode(result_idx=0)),
        ],
        edges=[
            MultiSourceEdge(source="root", target="yes", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="root", target="no", data=MultiEdgeData(sourceIndex=[1])),
        ],
        output=TreeOutput(data=[{"hit": 0}, {"hit": 1}], default={"hit": -1}, dtypes=[("hit", out_dtype)]),
    )


def _chain(conds, name="chain"):
    """`n0 -> n1 -> ... -> leaf_all (result n)`; node i's false edge -> leaf i."""
    nodes, edges, rows = [], [], []
    for i, cond in enumerate(conds):
        nodes.append(PositionedNode(id=f"n{i}", data=UnaryNode(condition=cond)))
        nodes.append(PositionedNode(id=f"leaf{i}", data=LeafNode(result_idx=i)))
        rows.append({"pts": i})
        edges.append(MultiSourceEdge(source=f"n{i}", target=f"leaf{i}", data=MultiEdgeData(sourceIndex=[1])))
        if i + 1 < len(conds):
            edges.append(MultiSourceEdge(source=f"n{i}", target=f"n{i + 1}", data=MultiEdgeData(sourceIndex=[0])))
    nodes.append(PositionedNode(id="leaf_all", data=LeafNode(result_idx=len(conds))))
    rows.append({"pts": len(conds)})
    edges.append(MultiSourceEdge(source=f"n{len(conds) - 1}", target="leaf_all", data=MultiEdgeData(sourceIndex=[0])))
    return Tree(name=name, nodes=nodes, edges=edges,
                output=TreeOutput(data=rows, default={"pts": -1}, dtypes=[("pts", "Int64")]))


# ---------------------------------------------------------------------------
# The defect, and its fix
# ---------------------------------------------------------------------------


def test_int64_above_2_53_is_compared_exactly_in_every_mode():
    """The verified defect: `[BIG, BIG + 1] == BIG` must answer `[1, 0]`.
    Through float64 both rows collapse to BIG and both match."""
    built = tree_module(_one_node(UnaryEqual(feature="n", threshold=BIG)), feature_types={"n": int})
    frame = pl.DataFrame({"n": pl.Series([BIG, BIG + 1], dtype=pl.Int64)})
    pipeline = flow(built.module)
    for mode in ("interpreted", "stepped", "fused"):
        assert pipeline.apply(frame, mode=mode)["hit"].to_list() == [1, 0], mode
    assert pipeline.score({"n": BIG + 1})["hit"] == 0
    assert pipeline.score({"n": BIG})["hit"] == 1
    assert_equivalent(pipeline, frame)


def test_an_undeclared_int_feature_is_still_float_and_still_collapses():
    """Honest negative control, pinned so nobody is surprised: with no
    declaration the feature is inferred `float` (the previous behaviour,
    unchanged for every existing document), and the collapse is still
    there. Declaring the kind is what fixes it, not merely upgrading."""
    built = tree_module(_one_node(UnaryEqual(feature="n", threshold=BIG), name="undeclared"))
    assert built.encoded.feature_kinds == {"n": "float"}
    frame = pl.DataFrame({"n": pl.Series([BIG, BIG + 1], dtype=pl.Int64)})
    assert flow(built.module).apply(frame)["hit"].to_list() == [1, 1]


def test_a_bool_feature_stays_a_bool_end_to_end():
    """`is_true`-only use infers `bool`: the path step declares the input
    `bool`, the boundary keeps the Boolean column as a bool array, the
    walker reads the bool row array. An Int64 0/1 column declared bool
    still works (the boundary's `astype(bool)`)."""
    built = tree_module(_one_node(UnaryIsTrue(feature="flag"), name="b"))
    assert built.encoded.feature_kinds == {"flag": "bool"}
    (inp,) = built.encoded.path_step.inputs
    assert inp.annotation is bool
    assert built.encoded.arrays["feat_kind"][0] == BOOL
    pipeline = flow(built.module)
    frame = pl.DataFrame({"flag": [True, False, True]})
    assert pipeline.apply(frame)["hit"].to_list() == [1, 0, 1]
    assert pipeline.score({"flag": False})["hit"] == 0
    assert_equivalent(pipeline, frame)
    ints = pl.DataFrame({"flag": pl.Series([1, 0, 7], dtype=pl.Int64)})
    assert pipeline.apply(ints)["hit"].to_list() == [1, 0, 1]


def test_an_int_feature_stays_an_int_end_to_end():
    built = tree_module(
        _chain([
            UnaryGreaterThan(feature="cents", threshold=100),
            UnaryBetween(feature="cents", min=150, max=250),
            UnaryIsIn(feature="cents", values=[200, 210]),
        ]),
        feature_types={"cents": "Int64"},
    )
    (inp,) = built.encoded.path_step.inputs
    assert inp.annotation is int
    assert all(p.annotation is int and isinstance(p.default, int) for p in built.encoded.path_step.params)
    assert set(built.encoded.arrays["feat_kind"]) <= {I64, F64}  # leaves carry 0
    pipeline = flow(built.module)
    frame = pl.DataFrame({"cents": pl.Series([50, 120, 200, 240, 300], dtype=pl.Int64)})
    assert pipeline.apply(frame)["pts"].to_list() == [0, 1, 3, 2, 1]
    assert_equivalent(pipeline, frame)


def test_a_mixed_tree_agrees_across_every_mode_and_explains_its_kinds():
    """float + int + bool + string_match + computed + isin, in one tree."""
    tree = _chain([
        UnaryLessThan(feature="score", threshold=700.0),
        UnaryGreaterThanEqual(feature="income_cents", threshold=5_000_00),
        UnaryIsTrue(feature="is_staff"),
        UnaryStringMatch(feature="sector", patterns=["retail", "public"]),
        UnaryGreaterThan(feature={"type": "computed", "expression": "score - burden"}, threshold=100.0),
        UnaryIsIn(feature="defaults", values=[0, 1]),
    ], name="mixed")
    built = tree_module(tree, feature_types={"income_cents": int, "defaults": int})
    assert built.encoded.feature_kinds == {
        "score": "float", "income_cents": "int", "is_staff": "bool", "sector": "str",
        "burden": "float", "defaults": "int",
    }
    report = built.explain()
    assert "income_cents: int" in report and "is_staff: bool" in report and "sector: str" in report
    # The matcher's int result rides in the int64 row array, not float64.
    matcher_input = next(i for i in built.encoded.path_step.inputs if i.name.endswith("__m_sector"))
    assert matcher_input.annotation is int
    pipeline = flow(built.module)
    # Row i fails at node i (its false edge -> leaf i); row 6 passes all.
    frame = pl.DataFrame({
        "score": [720.0, 650.0, 650.0, 650.0, 650.0, 650.0, 650.0],
        "income_cents": pl.Series([6_000_00, 4_000_00, 6_000_00, 6_000_00, 6_000_00, 6_000_00, 6_000_00], dtype=pl.Int64),
        "is_staff": [True, True, False, True, True, True, True],
        "sector": ["retail", "retail", "retail", "mining", "retail", "retail", "public"],
        "burden": [0.0, 0.0, 0.0, 0.0, 700.0, 0.0, 0.0],
        "defaults": pl.Series([0, 0, 0, 0, 0, 5, 1], dtype=pl.Int64),
    })
    assert pipeline.apply(frame)["pts"].to_list() == [0, 1, 2, 3, 4, 5, 6]
    assert_equivalent(pipeline, frame)


# ---------------------------------------------------------------------------
# What the single float64 array could never reject
# ---------------------------------------------------------------------------


def test_an_ordering_comparison_on_a_categorical_is_a_build_error():
    with pytest.raises(ValueError, match="no order"):
        encode_tree(_one_node(UnaryLessThan(feature="sector", threshold=5.0)), feature_types={"sector": str})
    # Inferred, too: a feature both string-matched and ordered is a str
    # feature being ordered.
    tree = _chain([
        UnaryStringMatch(feature="sector", patterns=["a"]),
        UnaryLessThan(feature="sector", threshold=5.0),
    ])
    with pytest.raises(ValueError, match="no order"):
        encode_tree(tree)


def test_other_meaningless_uses_are_build_errors():
    with pytest.raises(ValueError, match="Test a boolean with"):
        encode_tree(_one_node(UnaryEqual(feature="flag", threshold=1)), feature_types={"flag": bool})
    with pytest.raises(ValueError, match="no truth value"):
        encode_tree(_one_node(UnaryIsTrue(feature="sector")), feature_types={"sector": str})
    with pytest.raises(ValueError, match="needs a str column"):
        encode_tree(_one_node(UnaryStringMatch(feature="n", patterns=["a"])), feature_types={"n": int})
    with pytest.raises(ValueError, match="not an integer"):
        encode_tree(_one_node(UnaryLessThan(feature="n", threshold=5000.5)), feature_types={"n": int})
    with pytest.raises(ValueError, match="cannot read a typed"):
        encode_tree(
            _one_node(UnaryGreaterThan(feature={"type": "computed", "expression": "n - 1"}, threshold=0.0)),
            feature_types={"n": int},
        )
    with pytest.raises(ValueError, match="one kernel argument of one type"):
        encode_tree(
            _chain([
                UnaryLessThan(feature="a", threshold=InputRef(key="floor")),
                UnaryLessThan(feature="b", threshold=InputRef(key="floor")),
            ]),
            feature_types={"a": int, "b": float},
        )
    with pytest.raises(ValueError, match="does not read"):
        encode_tree(_one_node(UnaryLessThan(feature="x", threshold=1.0)), feature_types={"y": int})
    with pytest.raises(ValueError, match="not a recognised"):
        encode_tree(_one_node(UnaryLessThan(feature="x", threshold=1.0)), feature_types={"x": "Decimal"})


def test_polars_spellings_and_dtype_objects_are_accepted():
    tree = _chain([
        UnaryLessThan(feature="a", threshold=1),
        UnaryIsTrue(feature="b"),
        UnaryLessThan(feature="c", threshold=1.0),
    ])
    enc = encode_tree(tree, feature_types={"a": "Int64", "b": "Boolean", "c": pl.Float64})
    assert enc.feature_kinds == {"a": "int", "b": "bool", "c": "float"}


# ---------------------------------------------------------------------------
# Properties the migration must keep
# ---------------------------------------------------------------------------


def test_retuning_an_int_threshold_never_recompiles():
    built = tree_module(_one_node(UnaryGreaterThan(feature="cents", threshold=100), name="retune"),
                        feature_types={"cents": int})
    pipeline = flow(built.module)
    frame = pl.DataFrame({"cents": pl.Series([50, 150, 250], dtype=pl.Int64)})
    assert pipeline.apply(frame)["hit"].to_list() == [0, 1, 1]
    with count_new_compiles() as counted:
        out = pipeline.apply(frame, params={"retune": {"root_thr": 200}})
    assert out["hit"].to_list() == [0, 0, 1]
    assert counted.count == 0


def test_encoding_is_deterministic_and_carries_feat_kind():
    tree = _chain([UnaryLessThan(feature="a", threshold=1), UnaryIsTrue(feature="b")])
    first = encode_tree(tree, feature_types={"a": int})
    second = encode_tree(tree, feature_types={"a": int})
    assert first.arrays == second.arrays
    assert "feat_kind" in first.arrays
    assert first.feature_kinds == second.feature_kinds


def test_an_undeclared_tree_encodes_exactly_as_before():
    """No declaration, no `is_true`, no string: every feature float, every
    node reading the float64 array — the previous representation, slot
    for slot."""
    enc = encode_tree(_chain([UnaryLessThan(feature=f"f{i}", threshold=float(i)) for i in range(5)]))
    assert all(inp.annotation is float for inp in enc.path_step.inputs)
    assert set(enc.arrays["feat_kind"]) == {F64}
    assert enc.path_step.typed_args and enc.path_step.packed


def test_the_typed_layout_rule_is_slot_within_kind_in_input_order():
    step = Step(name="s", fn=None, inputs=(
        Input("a", float), Input("b", int), Input("c", float), Input("d", bool), Input("e", int),
    ), params=(), packed=True, typed_args=True)
    assert _typed_layout(step) == (
        FeatureKind.F64, FeatureKind.I64, FeatureKind.F64, FeatureKind.BOOL, FeatureKind.I64,
    )


# ---------------------------------------------------------------------------
# The reserved raw-string slot: real bytes, zero-copy, inside the kernel
# ---------------------------------------------------------------------------


def test_the_raw_string_slot_carries_polars_bytes_zero_copy():
    """A `bytes`-annotated input reads polars' own `offsets`/`values`
    buffers: the row array holds `(start, end)` and the byte buffer
    travels alongside, so a kernel can read the actual string. Proven with
    a step returning `len(s) * 1000 + first byte` — a lazy matcher (the
    sibling string strands) is this plus a comparison."""

    @njit(inline="always")
    def peek(args, params):
        f64, i64, b8, i32, s64, sbytes = args
        start, end = s64[0], s64[1]
        first = sbytes[0][start] if end > start else 0
        return (end - start) * 1000 + first + i64[0]

    step = Step(
        name="peek", fn=peek,
        inputs=(Input("s", bytes), Input("k", int)), params=(),
        packed=True, typed_args=True, output_annotation=int,
    )
    strings = ["ab", "", "xyz", "q"]
    series = pl.Series("s", strings)
    bufs = series._get_buffers()
    registry = {
        "s": bufs["offsets"].to_numpy(allow_copy=False),
        "__bytes__s": bufs["values"].to_numpy(allow_copy=False),
        "k": np.array([1, 2, 3, 4], dtype=np.int64),
    }
    expected = [len(s) * 1000 + (s.encode()[0] if s else 0) + k for s, k in zip(strings, [1, 2, 3, 4])]

    from decider2.compile.driver import _packed_input_arrays, ResolvedParams
    arrays = _packed_input_arrays(step, registry)
    out = np.empty(4, dtype=np.int64)
    build_packed_kernel(step)(arrays, ((), ()), 4, out)
    assert out.tolist() == expected
    # And the interpreted/stepped row path builds the same six-tuple.
    resolved = ResolvedParams(per_step_scalar={}, per_step_bundle={})
    rows = [peek(*_packed_row_args(step, None, registry, resolved, i)) for i in range(4)]
    assert rows == expected


# ---------------------------------------------------------------------------
# The deciding test: a typed tree's path_fn is a disk-cache HIT in a fresh
# process (doc 05 §4.2 condition 4), inline="always" notwithstanding.
# ---------------------------------------------------------------------------

_CHILD = textwrap.dedent(
    """
    import contextlib, io, json, warnings
    warnings.filterwarnings("ignore")
    import polars as pl
    from decider2 import flow
    from decider2.trees import (LeafNode, MultiEdgeData, MultiSourceEdge, PositionedNode, Tree,
                                TreeOutput, UnaryGreaterThan, UnaryIsTrue, UnaryNode, tree_module)

    tree = Tree(
        name="typed",
        nodes=[
            PositionedNode(id="a", data=UnaryNode(condition=UnaryGreaterThan(feature="cents", threshold=100))),
            PositionedNode(id="b", data=UnaryNode(condition=UnaryIsTrue(feature="flag"))),
            PositionedNode(id="l0", data=LeafNode(result_idx=0)),
            PositionedNode(id="l1", data=LeafNode(result_idx=1)),
            PositionedNode(id="l2", data=LeafNode(result_idx=2)),
        ],
        edges=[
            MultiSourceEdge(source="a", target="b", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="a", target="l0", data=MultiEdgeData(sourceIndex=[1])),
            MultiSourceEdge(source="b", target="l2", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="b", target="l1", data=MultiEdgeData(sourceIndex=[1])),
        ],
        output=TreeOutput(data=[{"pts": 0}, {"pts": 1}, {"pts": 2}], default={"pts": -1}, dtypes=[("pts", "Int64")]),
    )
    built = tree_module(tree, feature_types={"cents": int})
    frame = pl.DataFrame({"cents": pl.Series([50, 150, 150], dtype=pl.Int64), "flag": [True, False, True]})
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        pipe = flow(built.module)
        fused = pipe.apply(frame, mode="fused")["pts"].to_list()
        stepped = pipe.apply(frame, mode="stepped")["pts"].to_list()
    lines = log.getvalue().splitlines()
    print(json.dumps({
        "fused": fused, "stepped": stepped,
        "saved": [l.rsplit("/", 1)[-1] for l in lines if "[cache] data saved" in l],
        "loaded": [l.rsplit("/", 1)[-1] for l in lines if "[cache] data loaded" in l],
    }))
    """
)


def _run_child(tmp_path: Path) -> dict:
    src_dir = str(Path(decider2.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src_dir, env.get("PYTHONPATH")) if p)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "numba_cache")
    env["NUMBA_DEBUG_CACHE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.splitlines()[-1])


def test_a_typed_tree_path_fn_is_a_genuine_cache_hit_in_a_fresh_process(tmp_path):
    cold = _run_child(tmp_path)
    warm = _run_child(tmp_path)
    assert cold["fused"] == cold["stepped"] == warm["fused"] == warm["stepped"] == [0, 1, 2]
    assert sum(1 for n in cold["saved"] if "path_fn" in n) >= 1, cold
    assert sum(1 for n in warm["loaded"] if "path_fn" in n) >= 1, warm
    assert not any("path_fn" in n for n in warm["saved"]), warm
