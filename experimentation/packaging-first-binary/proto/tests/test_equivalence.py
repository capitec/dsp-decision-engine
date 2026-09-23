"""The two backends must give identical answers to a plain-Python reference
on every shape, and the facade must pick the backend the policy says.
Run: python -m pytest tests -q   (in a venv where d2shim is installed)
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings

import numpy as np
import polars as pl
import pytest

from d2shim import _arrowc as arrowc
from d2shim import pure, strings
from d2shim.pure import CONTAINS, EXACT, LEAF, PREFIX, STR, SUFFIX, build_tree

try:
    from d2shim.kernel_compiled import run_tree_compiled
    HAVE_COMPILED = True
except ImportError:
    HAVE_COMPILED = False

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
        yield "sliced-offset-3", s.slice(3, len(strings) - 5)


def both(s, tree, pattern):
    feats = np.empty((len(s), 0))
    got = {}
    v = arrowc.export(s)
    got["pure"] = pure.run_tree(feats, [v], tree, [pattern]).tolist()
    v.release()
    if HAVE_COMPILED:
        got["compiled"] = run_tree_compiled(feats, [s], tree, [pattern]).tolist()
        got["compiled-unchecked"] = run_tree_compiled(feats, [s], tree, [pattern], checked=False).tolist()
    return got


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("pattern", PATTERNS)
def test_matches_reference(mode, pattern):
    tree = one_node_tree(mode)
    for label, s in variants(STRINGS):
        expect = ref(s.to_list(), pattern, mode)
        for k, out in both(s, tree, pattern).items():
            assert out == expect, (label, mode, pattern, k)


@pytest.mark.parametrize("value", ["dog", "a much longer merchant descriptor", None, "", "exactly12chr"])
def test_single_record(value):
    for mode in MODES:
        s = pl.Series("s", [value], dtype=pl.String)
        for k, out in both(s, one_node_tree(mode), "dog").items():
            assert out == ref([value], "dog", mode), (value, mode, k)


def test_zero_rows_and_all_null():
    for k, out in both(pl.Series("s", [], dtype=pl.String), one_node_tree("exact"), "dog").items():
        assert out == [], k
    for k, out in both(pl.Series("s", [None, None, None], dtype=pl.String), one_node_tree("contains"), "").items():
        assert out == [0, 0, 0], k


def test_random_3000():
    rng = np.random.default_rng(0)
    words = ["dog", "cat", "a much longer merchant descriptor", "", "hotdog", "dogfood", "ünï dog"]
    strings = [None if rng.random() < 0.1 else words[rng.integers(len(words))] + ("x" * int(rng.integers(0, 40)))
               for _ in range(3000)]
    for mode in MODES:
        tree = one_node_tree(mode)
        for label, s in variants(strings):
            expect = ref(s.to_list(), "dog", mode)
            for k, out in both(s, tree, "dog").items():
                assert out == expect, (label, mode, k)


# ---- the facade's policy, each case in a fresh interpreter ------------------
PROBE = (
    "import warnings, json, d2shim; warnings.simplefilter('always');"
    "w=[];"
    "import warnings as _w; _w.showwarning=lambda m,c,f,l,*a: w.append(f'{c.__name__}: {m}');"
    "n,r=d2shim.which(); print(json.dumps({'backend':n,'reason':r,'warnings':w}))"
)


def _run(env_extra, hide_ext=False):
    env = {**os.environ, **env_extra}
    env.pop("D2SHIM_BACKEND", None) if "D2SHIM_BACKEND" not in env_extra else None
    code = PROBE
    if hide_ext:  # simulate a platform with no wheel: make the extension unimportable
        code = "import sys; sys.modules['d2shim._nashim'] = None; " + code
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    return p


def test_policy_auto_prefers_compiled():
    if not HAVE_COMPILED:
        pytest.skip("no compiled backend in this install")
    import json
    p = _run({})
    assert p.returncode == 0, p.stderr
    d = json.loads(p.stdout)
    assert d["backend"] == "compiled" and d["warnings"] == [], d


def test_policy_auto_falls_back_with_one_warning():
    import json
    p = _run({}, hide_ext=True)
    assert p.returncode == 0, p.stderr
    d = json.loads(p.stdout)
    assert d["backend"] == "pure"
    assert len(d["warnings"]) == 1 and d["warnings"][0].startswith("FallbackWarning"), d


def test_policy_forced_compiled_raises_when_missing():
    p = _run({"D2SHIM_BACKEND": "compiled"}, hide_ext=True)
    assert p.returncode != 0
    assert "ImportError" in p.stderr and "D2SHIM_BACKEND=compiled" in p.stderr, p.stderr


def test_policy_forced_pure_never_touches_extension():
    import json
    p = _run({"D2SHIM_BACKEND": "pure"})
    d = json.loads(p.stdout)
    assert d["backend"] == "pure" and d["warnings"] == [], d


def test_facade_runs_and_agrees():
    tree = one_node_tree("contains")
    s = pl.Series("s", STRINGS)
    expect = ref(STRINGS, "dog", "contains")
    assert strings.run_tree(np.empty((len(STRINGS), 0)), [s], tree, ["dog"]).tolist() == expect
    d = strings.diagnose()
    assert d["backend"] in ("compiled", "pure") and d["vendor_pin"].startswith("nanoarrow 0.9.0")
