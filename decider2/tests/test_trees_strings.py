"""Strings at the node — docs/BOUNDARY-REWORK.md §3.1, Stage 2.

A tree tests a string feature by its BYTES, at the node: `exact`,
`starts_with`, `ends_with` and `contains` against a pattern table that is
a kernel ARGUMENT. The feature crosses the boundary as a span (address,
length) into polars' own memory through the compiled Arrow shim; `score()`
builds the same span from `str.encode("utf-8")` with no polars at all, so
the fourth rung of `assert_equivalent` — `score` against `apply` — is the
assertion that nanoarrow's decode of a Utf8View element yields the bytes
Python's encoder yields (§1.6, §8).

The deciding tests: every match type agrees with a Python reference over
the string corpus of §8.2 (null, empty, 1/12/13 bytes, multi-byte UTF-8
with a multi-byte pattern, a pattern longer than the value, a pattern at
position 0 / at the end / absent, identical 12-byte prefixes); the frame
shapes of §8.2 give byte-identical output; all four rungs agree, the
`score()` rung included; editing, ADDING or removing a pattern compiles
nothing (§6); injected span drift fails at the `score` rung and nowhere
else; and `regex` / `case_sensitive=False` / `trim_whitespace` are still
refused by name.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from decider2 import flow
from decider2.testing import (
    assert_equivalent,
    assert_no_compilation_after_warmup,
    assert_no_recompile,
    corpus,
    count_new_compiles,
)
from decider2.testing.corpus import STRING_CORPUS
from decider2.trees import (
    InputRef,
    LeafNode,
    MultiEdgeData,
    MultiSourceEdge,
    PositionedNode,
    Tree,
    TreeOutput,
    UnaryGreaterThan,
    UnaryNode,
    UnaryStringMatch,
    UnsupportedInKernel,
    encode_tree,
    tree_module,
)
from decider2.trees.schema import CasesStringMatch, StringMatchCondition

MATCH_TYPES = ("exact", "starts_with", "ends_with", "contains")


def _one_node(cond, name="t"):
    """`root -> yes (hit 1) / no (hit 0)`."""
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
        output=TreeOutput(data=[{"hit": 0}, {"hit": 1}], default={"hit": -1}, dtypes=[("hit", "Int64")]),
    )


def _string_pipeline(match_type: str, patterns: list, name: str = "t"):
    built = tree_module(_one_node(UnaryStringMatch(feature="s", patterns=patterns, match_type=match_type), name=name))
    return flow(built.module)


_PY = {
    "exact": lambda v, p: v == p,
    "starts_with": lambda v, p: v.startswith(p),
    "ends_with": lambda v, p: v.endswith(p),
    "contains": lambda v, p: p in v,
}


def _reference(match_type: str, patterns: list[str], value: str | None) -> int | None:
    """Python's own `str` semantics. Valid for a byte matcher because UTF-8
    is self-synchronising: a code-point prefix/suffix/substring is exactly
    a byte prefix/suffix/substring of the encodings."""
    if value is None:
        return None
    return 1 if any(_PY[match_type](value, p) for p in patterns) else 0


# The values every string test is driven with (§8.2 item 2): the corpus
# plus null and empty.
VALUES: list[str | None] = [None, ""] + [v for _, v in STRING_CORPUS]

# Pattern sets that exercise: a multi-byte pattern, a pattern longer than
# any value, a pattern at position 0, at the end, absent, the empty
# pattern, and the 12-byte inline prefix of the two `prefix12_*` values.
PATTERN_SETS: list[list[str]] = [
    ["twelve chars"],                       # exact 12 bytes; a prefix of the two 13-byte siblings
    ["twelve charsA"],                      # tells the identical-prefix siblings apart at byte 12
    ["wörld", "本語の"],                     # multi-byte patterns
    ["🚀", "e"],                            # 4-byte sequence; a one-byte pattern
    ["z" * 200],                            # longer than every value
    ["thirteen", "har", "zzz"],             # position 0, the end, absent
    [""],                                   # empty: exact matches "" only; the others match everything
    ["a", "twelve charsB", "x" * 100],
]


# ---------------------------------------------------------------------------
# Acceptance 2: every match type, on a real polars String column, against
# the Python reference over the string corpus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("match_type", MATCH_TYPES)
@pytest.mark.parametrize("patterns", PATTERN_SETS, ids=[",".join(p)[:24] or "empty" for p in PATTERN_SETS])
def test_every_match_type_agrees_with_python_over_the_string_corpus(match_type, patterns):
    pipeline = _string_pipeline(match_type, patterns)
    frame = pl.DataFrame({"s": VALUES})
    expected = [_reference(match_type, patterns, v) for v in VALUES]
    out = pipeline.apply(frame)["hit"].to_list()
    assert out == expected, (match_type, patterns)
    # score() builds its span from Python bytes, never from polars
    assert [pipeline.score({"s": v})["hit"] for v in VALUES if v is not None] == [e for e in expected if e is not None]
    assert_equivalent(pipeline, frame)


def test_a_string_is_never_truncated_at_the_utf8view_inline_boundary():
    """The silent failure the corpus exists for: 12 bytes is the last
    inline Utf8View length, 13 the first out-of-line one. Two values that
    share a 12-byte prefix must not compare equal, and a 13-byte value must
    match its own 13-byte pattern in every mode."""
    exact = _string_pipeline("exact", ["twelve charsA"])
    frame = pl.DataFrame({"s": ["twelve charsA", "twelve charsB", "twelve chars", "thirteen char"]})
    assert exact.apply(frame)["hit"].to_list() == [1, 0, 0, 0]
    suffix = _string_pipeline("ends_with", ["sB"])
    assert suffix.apply(frame)["hit"].to_list() == [0, 1, 0, 0]
    assert_equivalent(exact, frame)
    assert_equivalent(suffix, frame)


# ---------------------------------------------------------------------------
# Acceptance 3: the frame-shape corpus gives byte-identical output
# ---------------------------------------------------------------------------


def test_frame_shapes_give_byte_identical_outputs():
    pipeline = _string_pipeline("contains", ["har", "🚀", "twelve"])
    fresh = pl.DataFrame({"s": VALUES, "k": list(range(len(VALUES)))})
    reference = pipeline.apply(fresh)["hit"].to_numpy()

    sliced = fresh[3:9]
    assert pipeline.apply(sliced)["hit"].to_numpy().tobytes() == reference[3:9].tobytes()
    sliced = fresh[1:]
    assert pipeline.apply(sliced)["hit"].to_numpy().tobytes() == reference[1:].tobytes()

    concat = pl.concat([fresh[:4], fresh[4:7], fresh[7:]], rechunk=False)
    assert concat.n_chunks() == 3
    assert pipeline.apply(concat)["hit"].to_numpy().tobytes() == reference.tobytes()

    widened = fresh.with_columns(pl.Series("extra", np.arange(len(VALUES), dtype=np.float64)))
    assert pipeline.apply(widened)["hit"].to_numpy().tobytes() == reference.tobytes()

    rechunked_str = fresh.with_columns(
        pl.concat([fresh["s"][:5], fresh["s"][5:]], rechunk=False).alias("s")
    )
    assert pipeline.apply(rechunked_str)["hit"].to_numpy().tobytes() == reference.tobytes()


def test_a_zero_row_frame_and_a_frame_with_every_row_routed_still_run():
    pipeline = _string_pipeline("exact", ["a"])
    frames = corpus(pipeline)
    assert pipeline.apply(frames["empty"])["hit"].to_list() == []
    all_null = pl.DataFrame({"s": pl.Series([None, None], dtype=pl.String)})
    assert pipeline.apply(all_null)["hit"].to_list() == [None, None]
    absent = pl.DataFrame({"other": [1.0, 2.0]})
    assert pipeline.apply(absent)["hit"].to_list() == [None, None]


# ---------------------------------------------------------------------------
# Acceptance 4: all four rungs, the score() rung included, over the corpus
# ---------------------------------------------------------------------------


class _CountingScore:
    """`assert_equivalent` sees a pipeline; this counts how many times it
    reached the fourth rung, so "score() is no longer skipped for string
    inputs" is a measured fact rather than an assumption."""

    def __init__(self, pipeline):
        self._pipeline = pipeline
        self.score_calls = 0

    def __getattr__(self, name):
        return getattr(self._pipeline, name)

    def score(self, *args, **kwargs):
        self.score_calls += 1
        return self._pipeline.score(*args, **kwargs)


@pytest.mark.parametrize("match_type", MATCH_TYPES)
def test_assert_equivalent_drives_all_four_rungs_over_the_generated_corpus(match_type):
    pipeline = _CountingScore(_string_pipeline(match_type, ["twelve", "charsA", "本語", "🚀🚀"]))
    frames = corpus(pipeline)
    boundary = frames["boundary"]
    assert any(c.startswith("str_thirteen_bytes") for c in boundary["case"].to_list())
    assert_equivalent(pipeline, boundary)
    assert pipeline.score_calls == boundary.height  # one score() per corpus row, none skipped


def test_a_mixed_tree_with_a_threshold_and_a_string_node_agrees_in_every_mode():
    tree = Tree(
        name="mixed",
        nodes=[
            PositionedNode(id="a", data=UnaryNode(condition=UnaryGreaterThan(feature="x", threshold=10.0))),
            PositionedNode(id="b", data=UnaryNode(condition=UnaryStringMatch(feature="s", patterns=["vip", "gold"], match_type="starts_with"))),
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
    pipeline = flow(tree_module(tree).module)
    frame = pl.DataFrame({"x": [5.0, 20.0, 20.0, 20.0], "s": ["vip", "vip-plus", "silver", "golden"]})
    assert pipeline.apply(frame)["pts"].to_list() == [0, 2, 1, 2]
    assert pipeline.score({"x": 20.0, "s": "goldfish"})["pts"] == 2
    assert_equivalent(pipeline, frame)
    assert_equivalent(pipeline, corpus(pipeline)["boundary"])


# ---------------------------------------------------------------------------
# Acceptance 5: a pattern is a value — editing, adding or removing one
# never recompiles
# ---------------------------------------------------------------------------


def test_editing_adding_and_removing_patterns_never_recompiles():
    pipeline = _string_pipeline("contains", ["ab"], name="retune")
    frame = pl.DataFrame({"s": ["ab", "xabx", "cd", "twelve charsA", "日本語"]})
    assert pipeline.apply(frame)["hit"].to_list() == [1, 1, 0, 0, 0]

    # edit the text (and its length across the 12/13-byte boundary)
    assert_no_recompile(pipeline, frame, {"retune": {"root_patterns": ["cd"]}},
                        {"retune": {"root_patterns": ["twelve charsA"]}})
    # ADD patterns: one -> three, including a multi-byte one
    assert_no_recompile(pipeline, frame, {"retune": {"root_patterns": ["ab"]}},
                        {"retune": {"root_patterns": ["ab", "cd", "本語"]}})
    # remove them all but one; a very long one
    assert_no_recompile(pipeline, frame, {"retune": {"root_patterns": ["ab", "cd", "本語"]}},
                        {"retune": {"root_patterns": ["q" * 5000]}})

    with count_new_compiles() as counted:
        out = pipeline.apply(frame, params={"retune": {"root_patterns": ["cd", "本語", "sA"]}})
        record = pipeline.score({"s": "日本語"}, params={"retune": {"root_patterns": ["cd", "本語", "sA"]}})
    assert out["hit"].to_list() == [0, 0, 1, 1, 1]
    assert record["hit"] == 1
    assert counted.count == 0


def test_a_pattern_count_change_and_a_threshold_change_share_one_kernel_signature():
    """Where today adding a literal changed a tuple's TYPE and forced a
    recompile (§6 item 3), the pattern table's numba type is fixed: the
    path step of a one-pattern tree and of a three-pattern tree compile
    against the identical signature."""
    one = encode_tree(_one_node(UnaryStringMatch(feature="s", patterns=["a"])))
    three = encode_tree(_one_node(UnaryStringMatch(feature="s", patterns=["a", "bb", "ccc"])))
    from decider2.compile.driver import _typed_params
    from numba import typeof

    sig_one = typeof(_typed_params(one.path_step, [p.default for p in one.path_step.params]))
    sig_three = typeof(_typed_params(three.path_step, [p.default for p in three.path_step.params]))
    assert sig_one == sig_three


def test_precompile_leaves_nothing_for_the_first_string_request_to_compile():
    pipeline = _string_pipeline("ends_with", ["ing"], name="warm")
    frame = pl.DataFrame({"s": ["running", "ran", "héllo wörld", "thirteen char"]})
    assert_no_compilation_after_warmup(
        pipeline,
        extra=[(frame, {}), ({"s": "singing"}, {}), ({"s": "日本語"}, {}),
               (frame, {"params": {"warm": {"root_patterns": ["語", "char"]}}})],
    )


# ---------------------------------------------------------------------------
# Acceptance 6: injected drift is caught at the score() rung — the
# independent producer (§1.6) — and nowhere else
# ---------------------------------------------------------------------------


def test_injected_span_drift_fails_between_apply_and_score_and_not_between_modes(monkeypatch):
    """interpreted/stepped/fused all read the SAME Arrow import, so a
    span corrupted at the boundary reaches all three identically and they
    still agree with each other. `score()` builds its span from Python's
    own bytes, so the corruption surfaces there — and only there."""
    from decider2.boundary.dtypes import SpanPlan
    from decider2.testing.equivalence import _assert_rung, _run_mode

    original = SpanPlan.extract

    def drifted(self, series):
        table, categories = original(self, series)
        table = table.copy()
        table[0, 1] -= 1   # row 0 silently loses its last byte at the boundary
        return table, categories

    monkeypatch.setattr(SpanPlan, "extract", drifted)

    pipeline = _string_pipeline("exact", ["hello"])
    frame = pl.DataFrame({"s": ["hello", "hello"]})
    outs = {mode: _run_mode(pipeline, frame, mode, {}) for mode in ("interpreted", "stepped", "fused")}
    assert outs["fused"][0]["hit"].to_list() == [0, 1]   # the drift is real, and shared
    _assert_rung("interpreted", *outs["interpreted"], "stepped", *outs["stepped"])   # agree
    _assert_rung("stepped", *outs["stepped"], "fused", *outs["fused"])               # agree
    with pytest.raises(AssertionError) as caught:
        assert_equivalent(pipeline, frame)
    message = str(caught.value)
    assert "between fused (batch) and score" in message and "row 0" in message
    assert "apply=0" in message and "score=1" in message


# ---------------------------------------------------------------------------
# Acceptance 7: what the kernel still cannot do is refused by name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"match_type": "regex"}, "regex engine"),
        ({"case_sensitive": False}, "to_lowercase"),
        ({"trim_whitespace": True}, "strip_chars"),
    ],
)
def test_regex_case_folding_and_trimming_are_still_refused(kwargs, expected):
    with pytest.raises(UnsupportedInKernel, match=expected):
        encode_tree(_one_node(UnaryStringMatch(feature="s", patterns=["a"], **kwargs)))
    cases = Tree(
        name="c",
        nodes=[
            PositionedNode(id="root", data=CasesStringMatch(
                feature="s", conditions=[StringMatchCondition(patterns=["a"])], **kwargs)),
            PositionedNode(id="l0", data=LeafNode(result_idx=0)),
            PositionedNode(id="l1", data=LeafNode(result_idx=-1)),
        ],
        edges=[
            MultiSourceEdge(source="root", target="l0", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="root", target="l1", data=MultiEdgeData(sourceIndex=[1])),
        ],
    )
    with pytest.raises(UnsupportedInKernel, match=expected):
        encode_tree(cases)


# ---------------------------------------------------------------------------
# The rest of the surface
# ---------------------------------------------------------------------------


def test_an_inputref_pattern_is_a_str_param_retuned_by_its_key():
    pipeline = _string_pipeline("starts_with", ["lit", InputRef(key="tier")], name="ref")
    frame = pl.DataFrame({"s": ["literal", "gold", "goldfish", "silver", ""]})
    assert pipeline.apply(frame)["hit"].to_list() == [1, 1, 1, 1, 1]   # default "" is a prefix of everything
    assert pipeline.apply(frame, params={"ref": {"tier": "gold"}})["hit"].to_list() == [1, 1, 1, 0, 0]
    assert pipeline.score({"s": "silverback"}, params={"ref": {"tier": "silver"}})["hit"] == 1
    assert_no_recompile(pipeline, frame, {"ref": {"tier": "gold"}}, {"ref": {"tier": "日本語のテキスト"}})
    assert_equivalent(pipeline, frame, params={"ref": {"tier": "gold"}})


def test_two_nodes_on_one_column_and_two_trees_on_one_column_keep_their_own_patterns():
    tree = Tree(
        name="two",
        nodes=[
            PositionedNode(id="a", data=UnaryNode(condition=UnaryStringMatch(feature="s", patterns=["x"], match_type="starts_with"))),
            PositionedNode(id="b", data=UnaryNode(condition=UnaryStringMatch(feature="s", patterns=["y"], match_type="ends_with"))),
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
    built = tree_module(tree)
    assert sorted(p.name for p in built.encoded.path_step.params) == ["a_patterns", "b_patterns"]
    pipeline = flow(built.module)
    frame = pl.DataFrame({"s": ["xy", "xz", "zy", "x", "y"]})
    assert pipeline.apply(frame)["pts"].to_list() == [2, 1, 0, 1, 0]
    assert pipeline.apply(frame, params={"two": {"b_patterns": ["z"]}})["pts"].to_list() == [1, 2, 0, 1, 0]
    assert_equivalent(pipeline, frame)

    other = tree_module(_one_node(UnaryStringMatch(feature="s", patterns=["z"], match_type="contains"), name="other"))
    both = flow(built.module, other.module)
    out = both.apply(frame)
    assert out["pts"].to_list() == [2, 1, 0, 1, 0] and out["hit"].to_list() == [0, 1, 1, 0, 0]
    assert_equivalent(both, frame)


def test_exact_over_many_patterns_is_an_isin():
    patterns = [f"code{i:03d}" for i in range(40)] + ["日本語"]
    pipeline = _string_pipeline("exact", patterns)
    values = ["code000", "code039", "code040", "code01", "日本語", "code039x"]
    assert pipeline.apply(pl.DataFrame({"s": values}))["hit"].to_list() == [1, 1, 0, 0, 1, 0]
    assert_equivalent(pipeline, pl.DataFrame({"s": values}))


def test_a_non_str_record_value_for_a_string_feature_is_a_type_error_naming_the_column():
    pipeline = _string_pipeline("exact", ["1"])
    with pytest.raises(TypeError, match="'s' is a string feature"):
        pipeline.score({"s": 1})


def test_a_null_string_routes_like_any_required_null():
    pipeline = _string_pipeline("contains", [""])   # "" is in every string; a null still never matches
    frame = pl.DataFrame({"s": ["a", None, ""]})
    assert pipeline.apply(frame)["hit"].to_list() == [1, None, 1]
    routed = pipeline.score({"s": None})
    assert routed["decision"] == "refer" and routed["routed_on"] == "s"


def test_a_categorical_column_is_refused_with_the_stage_6_hint_until_then():
    pipeline = _string_pipeline("exact", ["a"])
    frame = pl.DataFrame({"s": pl.Series(["a", "b"], dtype=pl.Categorical)})
    with pytest.raises(TypeError, match="dictionary"):
        pipeline.apply(frame)


def test_a_numeric_column_declared_as_a_string_feature_is_refused_not_misread():
    pipeline = _string_pipeline("exact", ["1"])
    with pytest.raises(TypeError, match="declared STR"):
        pipeline.apply(pl.DataFrame({"s": [1.0, 2.0]}))


def test_explain_reports_patterns_instead_of_matcher_steps():
    built = tree_module(_one_node(UnaryStringMatch(feature="s", patterns=["a", "bb"], match_type="contains")))
    report = built.explain()
    assert "string patterns : 2" in report and "string features : s" in report
    assert [s.name for s in built.module.steps] == ["t_path", "hit"]
    assert [i.name for i in flow(built.module).interface.inputs] == ["s"]
