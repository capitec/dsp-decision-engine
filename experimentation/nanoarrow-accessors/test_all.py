"""Correctness gate: A (incumbent), B (nanoarrow per row, unsafe and
checked), C (nanoarrow validate + numba, incumbent and trusting kernels)
must all equal a plain-Python reference on every shape before any timing
counts. Run: ../../.venv/bin/python -m pytest test_all.py -q
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import arrowc
from kernel import CONTAINS, EXACT, LEAF, PREFIX, STR, SUFFIX, build_tree, run_tree
from kernel_b import run_tree_b
from kernel_c import run_tree_c
from nashim import DEFAULT, FULL, MINIMAL, NanoView

MODES = {"exact": EXACT, "prefix": PREFIX, "suffix": SUFFIX, "contains": CONTAINS}
STRINGS = [
    "dog", "", None, "dogfood", "hotdog", "a dog in a long merchant descriptor", "exactly12chr",
    "thirteen chars", "dög", "ünïcödé dog ünïcödé", "DOG", None, "d", "do", "og",
    "a" * 100 + "dog", "dog" + "b" * 100, "x" * 50 + "dog" + "y" * 50,
]
PATTERNS = ["dog", "", "ö", "og", "a" * 100, "thirteen chars", "exactly12chr", "zzz"]


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
    return build_tree([(STR, 0, MODES[mode], 0, 1, 2, 0), (LEAF, 0, 0, 0, 0, 0, 1), (LEAF, 0, 0, 0, 0, 0, 0)])


def variants(strings):
    s = pl.Series("s", strings)
    yield "single-chunk", s
    if len(strings) >= 8:
        two = pl.concat([pl.Series("s", strings[:7]), pl.Series("s", strings[7:])])
        assert two.n_chunks() == 2
        yield "two-chunk", two
    if len(strings) >= 6:
        yield "sliced-offset-3", s.slice(3, len(strings) - 5)  # zero-copy slice: non-zero Arrow offset


def all_approaches(s, tree, pattern):
    n = len(s)
    feats = np.empty((n, 0))
    got = {}
    v = arrowc.export(s)
    got["A"] = run_tree(feats, [v], tree, [pattern]).tolist()
    v.release()
    nv = NanoView(s)
    got["B"] = run_tree_b(feats, [nv], tree, [pattern]).tolist()
    got["B-checked"] = run_tree_b(feats, [nv], tree, [pattern], checked=True).tolist()
    for lvl, name in ((MINIMAL, "min"), (DEFAULT, "def"), (FULL, "full")):
        got[f"C-{name}"] = run_tree_c(feats, [nv], tree, [pattern], level=lvl).tolist()
    got["C-trusting"] = run_tree_c(feats, [nv], tree, [pattern], level=FULL, trusting=True).tolist()
    nv.release()
    return got


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("pattern", PATTERNS)
def test_matches_reference(mode, pattern):
    tree = one_node_tree(mode)
    for label, s in variants(STRINGS):
        expect = ref(s.to_list(), pattern, mode)
        got = all_approaches(s, tree, pattern)
        for k, out in got.items():
            assert out == expect, (label, mode, pattern, k)


@pytest.mark.parametrize("value", ["dog", "a much longer merchant descriptor", None, "", "exactly12chr"])
def test_single_record(value):
    for mode in MODES:
        s = pl.Series("s", [value], dtype=pl.String)
        got = all_approaches(s, one_node_tree(mode), "dog")
        expect = ref([value], "dog", mode)
        for k, out in got.items():
            assert out == expect, (value, mode, k)


def test_zero_rows():
    s = pl.Series("s", [], dtype=pl.String)
    got = all_approaches(s, one_node_tree("exact"), "dog")
    for k, out in got.items():
        assert out == [], k


def test_all_null():
    s = pl.Series("s", [None, None, None], dtype=pl.String)
    got = all_approaches(s, one_node_tree("contains"), "")
    for k, out in got.items():
        assert out == [0, 0, 0], k


def test_random_mixed_columns():
    rng = np.random.default_rng(0)
    words = ["dog", "cat", "a much longer merchant descriptor", "", "hotdog", "dogfood", "ünï dog"]
    strings = [None if rng.random() < 0.1 else words[rng.integers(len(words))] + ("x" * int(rng.integers(0, 40)))
               for _ in range(3000)]
    for mode in MODES:
        tree = one_node_tree(mode)
        for label, s in variants(strings):
            expect = ref(s.to_list(), "dog", mode)
            got = all_approaches(s, tree, "dog")
            for k, out in got.items():
                assert out == expect, (label, mode, k)
