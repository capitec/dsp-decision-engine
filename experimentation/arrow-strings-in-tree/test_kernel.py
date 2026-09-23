"""Correctness: the STR node against a plain-Python reference.
Run: .venv/bin/python -m pytest experimentation/arrow-strings-in-tree/test_kernel.py -q
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

import arrowc
from kernel import CONTAINS, EXACT, LEAF, PREFIX, STR, SUFFIX, build_tree, run_tree

MODES = {"exact": EXACT, "prefix": PREFIX, "suffix": SUFFIX, "contains": CONTAINS}


def ref(strings, pattern, mode):
    out = []
    for s in strings:
        if s is None:
            out.append(0)
        elif mode == "exact":
            out.append(int(s == pattern))
        elif mode == "prefix":
            out.append(int(s.startswith(pattern)))
        elif mode == "suffix":
            out.append(int(s.endswith(pattern)))
        else:
            out.append(int(pattern in s))
    return out


def one_node_tree(mode):
    # node 0: STR col 0, pattern 0 -> leaf 1 else leaf 2
    return build_tree([
        (STR, 0, MODES[mode], 0, 1, 2, 0),
        (LEAF, 0, 0, 0, 0, 0, 1),
        (LEAF, 0, 0, 0, 0, 0, 0),
    ])


STRINGS = [
    "dog", "", None, "dogfood", "hotdog", "a dog in a long merchant descriptor", "exactly12chr",
    "thirteen chars", "dög", "ünïcödé dog ünïcödé", "DOG", None, "d", "do", "og",
    "a" * 100 + "dog", "dog" + "b" * 100, "x" * 50 + "dog" + "y" * 50,
]
PATTERNS = ["dog", "", "ö", "og", "a" * 100, "thirteen chars", "exactly12chr", "zzz"]


def _series_variants():
    s = pl.Series("s", STRINGS)
    yield "single-chunk", s
    two = pl.concat([pl.Series("s", STRINGS[:7]), pl.Series("s", STRINGS[7:])])
    assert two.n_chunks() == 2
    yield "two-chunk", two
    three = pl.concat([pl.Series("s", STRINGS[:2]), pl.Series("s", STRINGS[2:3]), pl.Series("s", STRINGS[3:])])
    assert three.n_chunks() == 3
    yield "three-chunk", three


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("pattern", PATTERNS)
def test_matches_reference(mode, pattern):
    for label, s in _series_variants():
        v = arrowc.export(s)
        assert v.format == "vu"
        out = run_tree(np.empty((len(STRINGS), 0)), [v], one_node_tree(mode), [pattern])
        assert out.tolist() == ref(STRINGS, pattern, mode), (label, mode, pattern)
        v.release()


def test_zero_rows():
    s = pl.Series("s", [], dtype=pl.String)
    v = arrowc.export(s)
    out = run_tree(np.empty((0, 0)), [v], one_node_tree("exact"), ["dog"])
    assert out.shape == (0,)


def test_all_null_column():
    s = pl.Series("s", [None, None, None], dtype=pl.String)
    v = arrowc.export(s)
    out = run_tree(np.empty((3, 0)), [v], one_node_tree("contains"), [""])
    assert out.tolist() == [0, 0, 0]  # null never matches, even the empty pattern


def test_chunked_reads_every_chunk_not_just_the_first():
    """The trap named in the brief: silently reading only chunk 0."""
    a = pl.Series("s", ["dog"] * 5)
    b = pl.Series("s", ["cat"] * 5)
    s = pl.concat([a, b])
    assert s.n_chunks() == 2
    v = arrowc.export(s)
    out = run_tree(np.empty((10, 0)), [v], one_node_tree("exact"), ["cat"])
    assert out.tolist() == [0] * 5 + [1] * 5


def test_from_dataframe_column_after_filter_and_concat():
    df = pl.concat([
        pl.DataFrame({"s": ["dog", "cat", None], "x": [1.0, 2.0, 3.0]}),
        pl.DataFrame({"s": ["dogma", "bird"], "x": [4.0, 5.0]}),
    ])
    s = df["s"]
    v = arrowc.export(s)
    out = run_tree(df["x"].to_numpy().reshape(-1, 1), [v], one_node_tree("prefix"), ["dog"])
    assert out.tolist() == [1, 0, 0, 1, 0]


def test_sliced_series_with_offset():
    """A polars slice is zero-copy and exports a chunk with a non-zero Arrow
    offset; the validity bitmap in particular must be read at bit offset+i."""
    base = pl.Series("s", ["x", None, "b", None, "c", "b", None, "d", "b", "b", "z"])
    for lo, hi in ((1, 11), (2, 9), (8, 11), (9, 10)):
        s = base[lo:hi]
        v = arrowc.export(s)
        assert any(c.offset != 0 for c in v.chunks), "expected a sliced chunk"
        out = run_tree(np.empty((hi - lo, 0)), [v], one_node_tree("exact"), ["b"])
        assert out.tolist() == ref(base.to_list()[lo:hi], "b", "exact"), (lo, hi)


def test_columns_with_different_chunk_layouts_are_refused_not_misaligned():
    two = pl.concat([pl.DataFrame({"s": ["dog", "cat"]}), pl.DataFrame({"s": ["hotdog", "dog"]})])
    df = two.with_columns(pl.Series("u", ["x", "dog", "y", "z"]))
    layouts = {n: df[n].n_chunks() for n in df.columns}
    if layouts["s"] == layouts["u"]:
        pytest.skip(f"polars aligned the chunks here: {layouts}")
    vs, vu = arrowc.export(df["s"]), arrowc.export(df["u"])
    tree = build_tree([(STR, 1, EXACT, 0, 1, 2, 0), (LEAF, 0, 0, 0, 0, 0, 1), (LEAF, 0, 0, 0, 0, 0, 0)])
    with pytest.raises(ValueError, match="chunk layouts"):
        run_tree(np.empty((4, 0)), [vs, vu], tree, ["dog"])
    # and after a rechunk it is right
    df2 = df.rechunk()
    vs, vu = arrowc.export(df2["s"]), arrowc.export(df2["u"])
    assert run_tree(np.empty((4, 0)), [vs, vu], tree, ["dog"]).tolist() == [0, 1, 0, 0]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
