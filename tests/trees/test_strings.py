"""String features: every match type, case folding and trimming, pattern params, in every mode."""
import re

import numpy as np
import polars as pl
import pytest
from numba import typeof

from decider import flow
from decider.engine import Engine
from decider.engine.compile import compile_call
from decider.engine.ir.context import to_ir
from decider.testing import no_recompile
from decider.steps.trees import TreeConfig

MATCH_TYPES = ("exact", "starts_with", "ends_with", "contains", "regex")
HIT = {"data": [{"hit": 0}, {"hit": 1}], "default": {"hit": -1}, "dtypes": [["hit", "Int64"]]}

# Values that trip a matcher reading bytes or prefixes wrongly: 1, 12 and 13
# bytes, multi-byte UTF-8, a long value, and two values sharing a 12-byte prefix.
CORPUS = ["a", "twelve chars", "thirteen char", "héllo wörld", "日本語のテキスト", "emoji 🚀🚀", "x" * 100,
          "twelve charsA", "twelve charsB"]
VALUES = [None, ""] + CORPUS

PATTERN_SETS = [
    ["twelve chars"],
    ["twelve charsA"],
    ["wörld", "本語の"],
    ["🚀", "e"],
    ["z" * 200],
    ["thirteen", "har", "zzz"],
    [""],
    ["a", "twelve charsB", "x" * 100],
]

_PY = {
    "exact": lambda v, p: v == p,
    "starts_with": str.startswith,
    "ends_with": str.endswith,
    "contains": lambda v, p: p in v,
    "regex": lambda v, p: re.search(p, v) is not None,
}


def _nodes(root: dict, name: str = "t", out: dict = HIT) -> TreeConfig:
    """`root -> yes (hit 1) / no (hit 0)`."""
    return TreeConfig(name=name, tree={
        "nodes": [{"id": "root", "data": root},
                  {"id": "yes", "data": {"type": "leaf", "result_idx": 1}},
                  {"id": "no", "data": {"type": "leaf", "result_idx": 0}}],
        "edges": [{"source": "root", "target": "yes", "data": {"sourceIndex": 0}},
                  {"source": "root", "target": "no", "data": {"sourceIndex": 1}}],
        "output": out,
    })


def _matcher(match_type: str, patterns: list, name: str = "t", **kw) -> TreeConfig:
    return _nodes({"type": "unary", "condition": {"op": "string_match", "feature": "s", "patterns": patterns,
                                                  "match_type": match_type, **kw}}, name)


def _expected(match_type: str, patterns: list[str], values: list, fold: bool = False, trim: bool = False) -> list:
    out = []
    for v in values:
        if v is None:
            out.append(0)
            continue
        v = v.strip() if trim else v
        ps = [p.lower() for p in patterns] if fold else patterns
        out.append(int(any(_PY[match_type](v.lower() if fold else v, p) for p in ps)))
    return out


@pytest.mark.parametrize("match_type", MATCH_TYPES)
@pytest.mark.parametrize("patterns", PATTERN_SETS, ids=[",".join(p)[:24] or "empty" for p in PATTERN_SETS])
def test_every_match_type_agrees_with_python_over_the_string_corpus(run, match_type, patterns):
    tree = _matcher(match_type, patterns)
    expected = _expected(match_type, patterns, VALUES)
    assert run(tree, pl.DataFrame({"s": VALUES}))["hit"].to_list() == expected
    exe = Engine().bind(tree, mode="fused")
    assert [exe.score({"s": v})["hit"] for v in VALUES] == expected


@pytest.mark.parametrize("kw", [{"case_sensitive": False}, {"trim_whitespace": True},
                                {"case_sensitive": False, "trim_whitespace": True}])
@pytest.mark.parametrize("match_type", MATCH_TYPES)
def test_case_folding_and_trimming_agree_with_python(run, match_type, kw):
    values = VALUES + ["  HELLO  ", "Hello", "WÖRLD wörld", " a"]
    patterns = ["hello", "Wörld", "a"]
    tree = _matcher(match_type, patterns, **kw)
    expected = _expected(match_type, patterns, values, not kw.get("case_sensitive", True),
                         kw.get("trim_whitespace", False))
    assert run(tree, pl.DataFrame({"s": values}))["hit"].to_list() == expected


def test_values_sharing_a_twelve_byte_prefix_are_told_apart(run):
    frame = pl.DataFrame({"s": ["twelve charsA", "twelve charsB", "twelve chars", "thirteen char"]})
    assert run(_matcher("exact", ["twelve charsA"]), frame)["hit"].to_list() == [1, 0, 0, 0]
    assert run(_matcher("ends_with", ["sB"]), frame)["hit"].to_list() == [0, 1, 0, 0]


def test_frame_shapes_give_identical_outputs(run):
    tree = _matcher("contains", ["har", "🚀", "twelve"])
    fresh = pl.DataFrame({"s": VALUES, "k": list(range(len(VALUES)))})
    reference = run(tree, fresh)["hit"].to_numpy()
    assert (run(tree, fresh[3:9])["hit"].to_numpy() == reference[3:9]).all()
    assert (run(tree, fresh[1:])["hit"].to_numpy() == reference[1:]).all()
    concat = pl.concat([fresh[:4], fresh[4:7], fresh[7:]], rechunk=False)
    assert concat.n_chunks() == 3
    assert (run(tree, concat)["hit"].to_numpy() == reference).all()
    widened = fresh.with_columns(pl.Series("extra", np.arange(len(VALUES), dtype=np.float64)))
    assert (run(tree, widened)["hit"].to_numpy() == reference).all()


def test_an_empty_frame_an_all_null_column_and_an_absent_column_still_run(run):
    tree = _matcher("exact", ["a"])
    assert run(tree, pl.DataFrame({"s": pl.Series([], dtype=pl.String)}))["hit"].to_list() == []
    assert run(tree, pl.DataFrame({"s": pl.Series([None, None], dtype=pl.String)}))["hit"].to_list() == [0, 0]
    assert run(tree, pl.DataFrame({"other": [1.0, 2.0]}))["hit"].to_list() == [0, 0]


def test_a_null_string_never_matches_not_even_the_empty_pattern(run):
    tree = _matcher("contains", [""])
    assert run(tree, pl.DataFrame({"s": ["a", None, ""]}))["hit"].to_list() == [1, 0, 1]
    assert Engine().bind(tree, mode="fused").score({"s": None})["hit"] == 0


def test_a_mixed_tree_with_a_threshold_and_a_string_node_agrees_in_every_mode(run):
    tree = TreeConfig(name="mixed", tree={
        "nodes": [
            {"id": "a", "data": {"type": "unary", "condition": {"op": ">", "feature": "x", "threshold": 10.0}}},
            {"id": "b", "data": {"type": "unary", "condition": {
                "op": "string_match", "feature": "s", "patterns": ["vip", "gold"], "match_type": "starts_with"}}},
            *[{"id": f"l{i}", "data": {"type": "leaf", "result_idx": i}} for i in range(3)],
        ],
        "edges": [{"source": "a", "target": "b", "data": {"sourceIndex": 0}},
                  {"source": "a", "target": "l0", "data": {"sourceIndex": 1}},
                  {"source": "b", "target": "l2", "data": {"sourceIndex": 0}},
                  {"source": "b", "target": "l1", "data": {"sourceIndex": 1}}],
        "output": {"data": [{"pts": 0}, {"pts": 1}, {"pts": 2}], "default": {"pts": -1},
                   "dtypes": [["pts", "Int64"]]},
    })
    frame = pl.DataFrame({"x": [5.0, 20.0, 20.0, 20.0], "s": ["vip", "vip-plus", "silver", "golden"]})
    assert run(tree, frame)["pts"].to_list() == [0, 2, 1, 2]
    assert Engine().bind(tree, mode="fused").score({"x": 20.0, "s": "goldfish"})["pts"] == 2


@pytest.mark.parametrize("match_type", ["exact", "starts_with", "ends_with", "contains"])
def test_byte_matches_compile_into_the_kernel(match_type):
    node = to_ir(_matcher(match_type, ["ab", {"param": "p", "default": "c"}]))
    assert [i.annotation for i in node.inputs] == [bytes | None]
    assert compile_call(node)[2] is None


@pytest.mark.parametrize("kw", [{"match_type": "regex"}, {"case_sensitive": False}, {"trim_whitespace": True}])
def test_regex_case_folding_and_trimming_run_the_python_walker_in_compiled_modes(kw):
    kw = {"match_type": "exact", **kw}
    assert compile_call(to_ir(_matcher(kw.pop("match_type"), ["ab"], **kw)))[2] is not None


def test_a_pattern_param_is_retuned_by_its_key(run):
    tree = _matcher("starts_with", ["lit", {"param": "tier", "default": ""}], name="ref")
    frame = pl.DataFrame({"s": ["literal", "gold", "goldfish", "silver", ""]})
    assert run(tree, frame)["hit"].to_list() == [1, 1, 1, 1, 1]   # "" is a prefix of everything
    assert run(tree, frame, {"ref": {"tier": "gold"}})["hit"].to_list() == [1, 1, 1, 0, 0]
    assert run(tree, frame, {"ref": {"tier": "日本語のテキスト"}})["hit"].to_list() == [1, 0, 0, 0, 0]
    exe = Engine().bind(tree, mode="fused")
    assert exe.score({"s": "silverback"}, params={"ref": {"tier": "silver"}})["hit"] == 1


def test_a_key_pattern_takes_its_default_from_the_parameters_block(run):
    tree = TreeConfig(name="ref", tree={
        "type": "flat_rule",
        "rule": {"rule": {"type": "unary", "condition": {"op": "string_match", "feature": "s",
                                                           "patterns": [{"key": "tier"}]},
                          "then": {"type": "leaf", "result_idx": 1}, "otherwise": {"type": "leaf", "result_idx": 0}}},
        "parameters": {"tier": {"type": "String", "default_value": "gold"}},
        "output": HIT,
    })
    frame = pl.DataFrame({"s": ["gold", "silver"]})
    assert run(tree, frame)["hit"].to_list() == [1, 0]
    assert run(tree, frame, {"ref": {"tier": "silver"}})["hit"].to_list() == [0, 1]


def test_two_nodes_on_one_column_and_two_trees_on_one_column_keep_their_own_patterns(run):
    tree = TreeConfig(name="two", tree={
        "nodes": [
            {"id": "a", "data": {"type": "unary", "condition": {
                "op": "string_match", "feature": "s", "patterns": ["x"], "match_type": "starts_with"}}},
            {"id": "b", "data": {"type": "unary", "condition": {
                "op": "string_match", "feature": "s", "patterns": [{"param": "b_pattern", "default": "y"}],
                "match_type": "ends_with"}}},
            *[{"id": f"l{i}", "data": {"type": "leaf", "result_idx": i}} for i in range(3)],
        ],
        "edges": [{"source": "a", "target": "b", "data": {"sourceIndex": 0}},
                  {"source": "a", "target": "l0", "data": {"sourceIndex": 1}},
                  {"source": "b", "target": "l2", "data": {"sourceIndex": 0}},
                  {"source": "b", "target": "l1", "data": {"sourceIndex": 1}}],
        "output": {"data": [{"pts": 0}, {"pts": 1}, {"pts": 2}], "default": {"pts": -1},
                   "dtypes": [["pts", "Int64"]]},
    })
    frame = pl.DataFrame({"s": ["xy", "xz", "zy", "x", "y"]})
    assert run(tree, frame)["pts"].to_list() == [2, 1, 0, 1, 0]
    assert run(tree, frame, {"two": {"b_pattern": "z"}})["pts"].to_list() == [1, 2, 0, 1, 0]
    both = flow(tree, _matcher("contains", ["z"], name="other"))
    out = run(both, frame)
    assert out["pts"].to_list() == [2, 1, 0, 1, 0] and out["hit"].to_list() == [0, 1, 1, 0, 0]


def test_exact_over_many_patterns_is_set_membership(run):
    tree = _matcher("exact", [f"code{i:03d}" for i in range(40)] + ["日本語"])
    values = ["code000", "code039", "code040", "code01", "日本語", "code039x"]
    assert run(tree, pl.DataFrame({"s": values}))["hit"].to_list() == [1, 1, 0, 0, 1, 0]


@pytest.mark.parametrize("kw", [{"match_type": "regex"}, {"case_sensitive": False}, {"trim_whitespace": True}])
def test_cases_string_match_folds_trims_and_matches_regexes(run, kw):
    patterns = {"match_type": "regex", "p": "^a.c$"} if "match_type" in kw else {"p": "abc"}
    tree = TreeConfig(name="c", tree={
        "nodes": [{"id": "root", "data": {"type": "cases", "op": "string_match", "feature": "s",
                                          "conditions": [{"patterns": [patterns["p"]]}], **kw}},
                  {"id": "l0", "data": {"type": "leaf", "result_idx": 0}}],
        "edges": [{"source": "root", "target": "l0", "data": {"sourceIndex": [0]}}],
        "output": {"data": [{"r": "hit"}], "default": {"r": "miss"}, "dtypes": [["r", "String"]]},
    })
    values = ["abc", "ABC", " abc ", "axc", "zzz"]
    got = run(tree, pl.DataFrame({"s": values}))["r"].to_list()
    expected = {
        "match_type": ["hit", "miss", "miss", "hit", "miss"],
        "case_sensitive": ["hit", "hit", "miss", "miss", "miss"],
        "trim_whitespace": ["hit", "miss", "hit", "miss", "miss"],
    }[next(iter(kw))]
    assert got == expected


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_a_long_string_column_is_read_in_place_and_an_override_replaces_it(mode):
    frame = pl.DataFrame({"s": ["gold card", "silver", None] * 20})
    exe = Engine().bind(flow(_matcher("starts_with", ["gold"])), mode=mode)
    assert exe.run(frame)["hit"].to_list() == [1, 0, 0] * 20
    s = exe.session(frame)
    s.set("s", "gold")
    s.resume()
    assert s.output()["hit"].to_list() == [1] * 60


def _gated(patterns: list, threshold: float = 10.0) -> TreeConfig:
    """`x > threshold` and `s` contains a pattern: hit 1, else 0."""
    return _nodes({"type": "composite", "op": "and", "conditions": [
        {"op": ">", "feature": "x", "threshold": threshold},
        {"op": "string_match", "feature": "s", "patterns": patterns, "match_type": "contains"}]}, name="retune")


def test_editing_adding_and_removing_patterns_never_recompiles():
    # Literal patterns are data the walker reads, so a document with other patterns reuses the kernels.
    frame = pl.DataFrame({"x": [20.0] * 5, "s": ["ab", "xabx", "cd", "twelve charsA", "日本語"]})
    for mode in ("stepped", "fused"):
        exe = Engine().bind(_gated(["ab"]), mode=mode)
        exe.run(frame)
        exe.score({"x": 20.0, "s": "ab"})
    edits = [(["cd"], [0, 0, 1, 0, 0]), (["twelve charsA"], [0, 0, 0, 1, 0]),
             (["ab", "cd", "本語"], [1, 1, 1, 0, 1]), (["q" * 5000], [0, 0, 0, 0, 0]),
             (["cd", "本語", "sA"], [0, 0, 1, 1, 1])]
    with no_recompile():
        for patterns, hits in edits:
            for mode in ("stepped", "fused"):
                exe = Engine().bind(_gated(patterns), mode=mode)
                assert exe.run(frame)["hit"].to_list() == hits
                assert [exe.score(r)["hit"] for r in frame.iter_rows(named=True)] == hits
    exe = Engine().bind(_gated(["ab", {"param": "extra", "default": "zz"}]), mode="fused")
    exe.run(frame)
    exe.score({"x": 20.0, "s": "cd"})
    with no_recompile():
        assert exe.run(frame, {"retune": {"extra": "語"}})["hit"].to_list() == [1, 1, 0, 0, 1]
        assert exe.score({"x": 20.0, "s": "cd"}, {"retune": {"extra": "d"}})["hit"] == 1


def test_a_pattern_count_change_and_a_threshold_change_share_one_kernel_signature():
    def signature(tree):
        node = to_ir(tree)
        return typeof(tuple(v for _, v in node.consts)), node.params

    assert signature(_gated(["a"])) == signature(_gated(["a", "bb", "ccc"])) == signature(_gated(["a"], 99.5))
