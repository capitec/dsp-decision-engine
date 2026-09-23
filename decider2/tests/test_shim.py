"""Stage 1 of the boundary rework (docs/BOUNDARY-REWORK.md §7 "Stage 1"):
the compiled nanoarrow shim, `FramePlan`/`FrameView`, packaging and the
cache proof — with NO consumer. Nothing in `decider2/src/` outside
`decider2/_arrow/` is touched by this stage, so these tests are the whole
evidence that the riskiest assumption in the plan holds: a compiled shim
can be built, shipped, loaded and called from a cached numba kernel inside
the real package, on a whole-frame import that handles slicing, nulls and
dictionaries.

Every string case here is one that produces a silent wrong answer if the
layout is misread: the 12/13-byte Utf8View inline boundary, the empty
string, a null, multi-byte UTF-8, a sliced frame (`df[a:b]` carries its
Arrow offset on each child) and a `pl.concat` multi-chunk frame.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numba import njit

import decider2
import decider2._arrow as arrow
from decider2._arrow import (
    GATHER_ADDR,
    GET_STRING_ADDR,
    ArrowKindError,
    ColumnSpec,
    FramePlan,
    FrameView,
    load_f64,
    load_i64,
    load_u8,
    string_lengths,
)
from decider2.types import FeatureKind as K

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# The exact cases that misread silently: inline (<= 12 bytes) vs out-of-line
# (>= 13 bytes) Utf8View elements, empty, null, multi-byte, long.
STRINGS = [
    "",                    # empty: length 0, inline
    "a",
    "twelve chars",        # 12 bytes: the LAST inline length
    "thirteen char",       # 13 bytes: the FIRST out-of-line length
    "x" * 12,
    "y" * 13,
    None,                  # null: validity bit clear
    "héllo wörld",         # 13 bytes of UTF-8 for 11 code points: out of line
    "日本語のテキスト",         # 3 bytes per code point
    "emoji 🚀🚀",           # 4-byte sequences
    "a" * 100,
    "z",
]
assert len("twelve chars".encode()) == 12 and len("thirteen char".encode()) == 13
assert len("héllo wörld".encode()) == 13


def _encoded(values):
    return [None if v is None else v.encode("utf-8") for v in values]


def _plan(df, kinds: dict, fills: dict | None = None) -> FramePlan:
    return FramePlan([ColumnSpec(n, k, (fills or {}).get(n)) for n, k in kinds.items()], df.columns)


def _view(df, kinds: dict, fills: dict | None = None) -> FrameView:
    return FrameView(_plan(df, kinds, fills)).bind(df)


def _nullable(values, valid):
    """Python-level view of a gathered column: None where the validity bit was clear."""
    return [v if ok else None for v, ok in zip(values, valid)]


# ---------------------------------------------------------------------------
# strings round-trip against Python's own str.encode("utf-8")
# ---------------------------------------------------------------------------

def test_strings_round_trip_including_the_inline_boundary_empty_null_and_multibyte():
    df = pl.DataFrame({"s": STRINGS})
    with _view(df, {"s": K.STR}) as fv:
        assert fv.describe()[0]["arrow_type"] == "string_view"       # polars 1.41 exports Utf8View
        assert fv.strings(0) == _encoded(STRINGS)                     # every row, via the cached kernel
        for i, s in enumerate(STRINGS):                               # and one row at a time
            fv.gather(i)
            assert fv.string(0) == (None if s is None else s.encode("utf-8"))
            assert fv.span[1] == (-1 if s is None else len(s.encode("utf-8")))
            assert fv.valid[0] == (0 if s is None else 1)


def test_get_string_through_the_accessor_pointer_matches_lengths_and_nulls():
    df = pl.DataFrame({"s": STRINGS})
    with _view(df, {"s": K.STR}) as fv:
        out = np.empty(fv.n, np.int64)
        string_lengths(np.uint64(GET_STRING_ADDR), np.uint64(fv.child_view("s")), fv.n, out)
    assert out.tolist() == [-1 if s is None else len(s.encode()) for s in STRINGS]


@pytest.mark.parametrize("a,b", [(0, 6), (1, 9), (3, 12), (7, 12)])
def test_a_sliced_frame_carries_its_offset_on_each_child_and_reads_correctly(a, b):
    n = len(STRINGS)
    df = pl.DataFrame({
        "s": STRINGS,
        "f": [None if i % 4 == 1 else i * 1.5 for i in range(n)],
        "i": [None if i % 5 == 2 else i * 1_000_000_007 for i in range(n)],
        "b": [None if i % 3 == 0 else (i % 2 == 0) for i in range(n)],
    })
    sliced = df[a:b]
    with _view(sliced, {"s": K.STR, "f": K.F64, "i": K.I64, "b": K.BOOL}) as fv:
        assert fv.n == b - a
        assert [d["offset"] for d in fv.describe()] == [a] * 4        # the struct is 0, the children carry it
        rows = fv.materialize()
        assert rows.strings(0) == _encoded(STRINGS[a:b])
        assert _nullable(rows.f64[:, 0].tolist(), rows.valid[:, 1]) == df["f"].to_list()[a:b]
        assert _nullable(rows.i64[:, 0].tolist(), rows.valid[:, 2]) == df["i"].to_list()[a:b]
        assert _nullable(rows.b8[:, 0].tolist(), rows.valid[:, 3]) == df["b"].to_list()[a:b]


def test_a_multi_chunk_frame_imports_as_one_struct_and_polars_rechunks_the_caller():
    a = pl.DataFrame({"s": ["a", None], "f": [1.0, 2.0]})
    b = pl.DataFrame({"s": ["bb" * 7, "c"], "f": [3.0, None]})
    df = pl.concat([a, b]).with_columns(pl.Series("z", [1, 2, 3, 4]))
    assert df.n_chunks("all") == [2, 2, 1]                            # columns with DIFFERENT chunk layouts
    with _view(df, {"s": K.STR, "f": K.F64, "z": K.I64}) as fv:
        assert (fv.n_chunks, fv.n) == (1, 4)                          # ONE aligned struct chunk arrived
        rows = fv.materialize()
        assert rows.strings(0) == [b"a", None, b"bb" * 7, b"c"]
        assert _nullable(rows.f64[:, 0].tolist(), rows.valid[:, 1]) == [1.0, 2.0, 3.0, None]
        assert rows.i64[:, 0].tolist() == [1, 2, 3, 4]
    # Pinned side-effect (BOUNDARY-REWORK.md §1.4): polars rechunked the
    # caller's frame IN PLACE while exporting. If a polars upgrade stops
    # doing this, the multi-chunk path in FrameView._pull_remaining_chunks
    # starts carrying real traffic and this assertion says so.
    assert df.n_chunks("all") == [1, 1, 1]


def test_a_zero_row_frame_binds_with_n_zero():
    df = pl.DataFrame({"s": pl.Series([], dtype=pl.String), "f": pl.Series([], dtype=pl.Float64)})
    with _view(df, {"s": K.STR, "f": K.F64}) as fv:
        assert fv.n == 0
        rows = fv.materialize()
        assert rows.f64.shape == (0, 1) and rows.strings(0) == []


def test_a_million_long_strings_span_more_than_one_variadic_buffer():
    n = 1_000_000
    values = [f"merchant descriptor {i:07d}" for i in range(n)]       # 27 bytes: never inline
    df = pl.DataFrame({"s": values})
    with _view(df, {"s": K.STR}) as fv:
        desc = fv.describe()[0]
        assert desc["n_variadic_buffers"] > 1, desc
        rows = fv.materialize()
        assert (rows.span[:, 1] == 27).all()
        for i in (0, 1, 4095, 4096, n // 2, 123_456, n - 1):
            assert ctypes.string_at(int(rows.span[i, 0]), 27) == values[i].encode()


# ---------------------------------------------------------------------------
# every row of §1.2's dtype table
# ---------------------------------------------------------------------------

def test_every_dtype_of_the_ladder_lands_in_its_kind_without_a_cast():
    df = pl.DataFrame({
        "f64": pl.Series([1.5, None], dtype=pl.Float64),
        "f32": pl.Series([2.5, None], dtype=pl.Float32),
        "i8": pl.Series([-128, None], dtype=pl.Int8),
        "i16": pl.Series([-32768, None], dtype=pl.Int16),
        "i32": pl.Series([-(2 ** 31), None], dtype=pl.Int32),
        "i64": pl.Series([2 ** 62 + 1, None], dtype=pl.Int64),            # above 2**53: no float round trip
        "u8": pl.Series([255, None], dtype=pl.UInt8),
        "u16": pl.Series([65535, None], dtype=pl.UInt16),
        "u32": pl.Series([2 ** 32 - 1, None], dtype=pl.UInt32),
        "u64": pl.Series([2 ** 63 - 1, None], dtype=pl.UInt64),
        "bool": pl.Series([True, None], dtype=pl.Boolean),
        "date": pl.Series([date(2024, 2, 29), None], dtype=pl.Date),
        "dt": pl.Series([datetime(2024, 2, 29, 12, 34, 56, 789), None], dtype=pl.Datetime("us")),
        "dur": pl.Series([timedelta(days=1, seconds=5), None], dtype=pl.Duration("ms")),
        "time": pl.Series([time(12, 34, 56), None], dtype=pl.Time),
    })
    kinds = {c: K.F64 for c in ("f64", "f32")}
    kinds.update({c: K.I64 for c in ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
                                     "date", "dt", "dur", "time")})
    kinds["bool"] = K.BOOL
    with _view(df, kinds) as fv:
        desc = {d["name"]: d for d in fv.describe()}
        assert desc["f64"]["arrow_type"] == "double" and desc["f32"]["arrow_type"] == "float"
        assert desc["i64"]["arrow_type"] == "int64" and desc["u64"]["arrow_type"] == "uint64"
        assert desc["bool"]["arrow_type"] == "bool"
        # temporal types: their STORAGE is the integer the kernel loads
        assert desc["date"]["storage_type"] == "int32"
        assert desc["dt"]["storage_type"] == "int64"
        assert desc["dur"]["storage_type"] == "int64"
        assert desc["time"]["storage_type"] == "int64"
        fv.gather(0)
        assert fv.f64.tolist() == [1.5, 2.5]
        i64_cols = [c for c, k in kinds.items() if k is K.I64]
        assert fv.i64.tolist() == [int(df[c].to_physical()[0]) for c in i64_cols]
        assert fv.i64[i64_cols.index("i64")] == 2 ** 62 + 1
        assert fv.b8.tolist() == [True]
        assert fv.valid.tolist() == [1] * len(kinds)
        fv.gather(1)
        assert fv.valid.tolist() == [0] * len(kinds)
        assert all(np.isnan(fv.f64)) and fv.i64.tolist() == [0] * len(i64_cols) and fv.b8.tolist() == [False]


def test_nulls_per_kind_take_the_default_or_the_declared_fill():
    df = pl.DataFrame({
        "f": [None, 1.0], "i": [None, 2], "b": [None, True],
        "c": pl.Series([None, "k"], dtype=pl.Categorical), "s": [None, "s"],
    })
    kinds = {"f": K.F64, "i": K.I64, "b": K.BOOL, "c": K.CODE, "s": K.STR}
    with _view(df, kinds) as fv:
        fv.gather(0)
        assert np.isnan(fv.f64[0]) and fv.i64[0] == 0 and fv.b8[0] == False and fv.i32[0] == -1  # noqa: E712
        assert fv.span.tolist() == [0, -1] and fv.valid.tolist() == [0] * 5
        fv.gather(1)
        assert fv.valid.tolist() == [1] * 5 and fv.string(0) == b"s"
    with _view(df, kinds, fills={"f": -1.0, "i": 7, "b": True, "c": 99}) as fv:
        fv.gather(0)
        assert (fv.f64[0], fv.i64[0], fv.b8[0], fv.i32[0]) == (-1.0, 7, True, 99)
        assert fv.valid.tolist() == [0] * 5                           # the fill does not fake validity
    with pytest.raises(ValueError, match="STR feature has no fill"):
        _plan(df, kinds, fills={"s": "x"})


def test_categorical_and_enum_expose_a_dictionary_view_of_the_right_length_and_index_type():
    df = pl.DataFrame({
        "c": pl.Series(["x", "y", None, "x"], dtype=pl.Categorical),
        "e": pl.Series(["lo", "hi", "lo", None], dtype=pl.Enum(["lo", "hi", "mid"])),
    })
    with _view(df, {"c": K.CODE, "e": K.CODE}) as fv:
        d = {x["name"]: x for x in fv.describe()}
        assert d["c"]["arrow_type"] == "dictionary(uint32)<string_view>"
        assert d["c"]["storage_type"] == "uint32" and d["c"]["dictionary_length"] == 2
        assert d["e"]["storage_type"] == "uint8" and d["e"]["dictionary_length"] == 3
        assert d["c"]["dictionary_storage_type"] == d["e"]["dictionary_storage_type"] == "string_view"
        rows = fv.materialize()
        assert rows.i32[:, 0].tolist() == [c if c is not None else -1 for c in df["c"].to_physical().to_list()]
        assert rows.i32[:, 1].tolist() == [0, 1, 0, -1]


# ---------------------------------------------------------------------------
# what is refused, and how it says so
# ---------------------------------------------------------------------------

def test_a_decimal_column_is_refused_with_the_frame_tier_cast_hint():
    df = pl.DataFrame({"d": pl.Series([Decimal("1.25")], dtype=pl.Decimal(10, 2))})
    fv = FrameView(FramePlan([("d", K.F64)], df.columns))
    with pytest.raises(ArrowKindError, match=r"column 'd' is Arrow decimal128\(10, 2\) but is declared F64.*frame tier"):
        fv.bind(df)
    assert not fv.bound                                               # the failed bind released the frame


def test_nested_columns_are_reported_not_crashed():
    df = pl.DataFrame({"l": [[1, 2], [3]], "st": [{"a": 1}, {"a": 2}]})
    # polars exports List as Arrow large_list; nanoarrow names it without its child
    with pytest.raises(ArrowKindError, match=r"column 'l' is Arrow (large_)?list.*KernelSplitPlan"):
        FrameView(FramePlan([("l", K.I64)], df.columns)).bind(df)
    with pytest.raises(ArrowKindError, match=r"column 'st' is Arrow struct.*KernelSplitPlan"):
        FrameView(FramePlan([("st", K.F64)], df.columns)).bind(df)


@pytest.mark.parametrize("column,kind,pattern", [
    ("f", K.I64, r"column 'f' is Arrow double but is declared I64"),          # the frame-tier cast case
    ("s", K.I64, r"column 's' is Arrow string_view but is declared I64"),
    ("s", K.CODE, r"declared CODE.*CODE is a dictionary index"),
    ("c", K.STR, r"dictionary\(uint32\)<string_view> but is declared STR.*Stage 6"),
    ("c", K.I64, r"dictionary\(uint32\)<string_view> but is declared I64"),
    ("b", K.F64, r"column 'b' is Arrow bool but is declared F64"),
])
def test_a_kind_mismatch_names_the_column_the_arrow_type_and_the_kind(column, kind, pattern):
    df = pl.DataFrame({"f": [1.0], "s": ["a"], "c": pl.Series(["a"], dtype=pl.Categorical), "b": [True]})
    with pytest.raises(ArrowKindError, match=pattern):
        FrameView(FramePlan([(column, kind)], df.columns)).bind(df)


def test_frame_view_misuse_is_an_error_not_a_wrong_answer():
    df = pl.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(KeyError, match="no column"):
        FramePlan([("zz", K.F64)], df.columns)
    fv = FrameView(FramePlan([("a", K.F64)], df.columns))
    with pytest.raises(RuntimeError, match="not bound"):
        fv.gather(0)
    with pytest.raises(ValueError, match="differ from the plan"):
        fv.bind(df.select("b", "a"))                                  # same columns, different order
    fv.bind(df)
    with pytest.raises(IndexError):
        fv.gather(1)
    fv.bind(pl.DataFrame({"a": [5.0], "b": [6.0]}))                   # rebinding releases the previous frame
    fv.gather(0)
    assert fv.f64[0] == 5.0
    fv.release()
    fv.release()                                                      # idempotent


# ---------------------------------------------------------------------------
# the address table: numba loads straight from the resolved (data, validity,
# offset) triples, no per-row C call (the design's V3 shape)
# ---------------------------------------------------------------------------

@njit
def _read_columns(addrs, n, out_f, out_i, out_b, valid_f):
    eight = np.uint64(8)
    three = np.uint64(3)
    seven = np.uint64(7)
    for i in range(n):
        u = np.uint64(i)
        out_f[i] = load_f64(addrs[0, 0] + eight * u)
        out_i[i] = load_i64(addrs[1, 0] + eight * u)
        j = addrs[2, 2] + u                                           # bool data is bit-indexed at offset + i
        out_b[i] = (load_u8(addrs[2, 0] + (j >> three)) >> (j & seven)) & np.uint8(1)
        v = addrs[0, 1]
        if v == np.uint64(0):
            valid_f[i] = 1
        else:
            jj = addrs[0, 2] + u
            valid_f[i] = (load_u8(v + (jj >> three)) >> (jj & seven)) & np.uint8(1)


@pytest.mark.parametrize("a,b", [(0, 8), (3, 8), (5, 7)])
def test_numba_loads_from_the_resolved_addresses_agree_with_polars(a, b):
    df = pl.DataFrame({
        "f": [None if i == 4 else float(i) for i in range(8)],
        "i": [i * 3 for i in range(8)],
        "b": [i % 3 == 0 for i in range(8)],
    })[a:b]
    with _view(df, {"f": K.F64, "i": K.I64, "b": K.BOOL}) as fv:
        n = fv.n
        out_f, out_i = np.empty(n, np.float64), np.empty(n, np.int64)
        out_b, valid_f = np.empty(n, np.uint8), np.empty(n, np.uint8)
        _read_columns(fv.addrs, n, out_f, out_i, out_b, valid_f)
    assert _nullable(out_f.tolist(), valid_f) == df["f"].to_list()
    assert out_i.tolist() == df["i"].to_list()
    assert [bool(x) for x in out_b] == df["b"].to_list()


# ---------------------------------------------------------------------------
# the deciding test: a cache=True kernel that calls the shim through a
# pointer ARGUMENT is a genuine disk-cache hit in a fresh process
# ---------------------------------------------------------------------------

_CACHE_CHILD = textwrap.dedent(
    """
    import contextlib, io, json, warnings
    warnings.filterwarnings("ignore")
    import numpy as np, polars as pl
    from decider2._arrow import FramePlan, FrameView, GATHER_ADDR, GET_STRING_ADDR, string_lengths
    from decider2.types import FeatureKind as K
    df = pl.DataFrame({"s": ["dog", None, "a much longer merchant descriptor"], "f": [1.0, None, 3.0]})
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        fv = FrameView(FramePlan([("s", K.STR), ("f", K.F64)], df.columns)).bind(df)
        rows = fv.materialize()
        lens = np.empty(fv.n, np.int64)
        string_lengths(np.uint64(GET_STRING_ADDR), np.uint64(fv.child_view("s")), fv.n, lens)
        strings = rows.strings(0)
        fv.release()
    lines = log.getvalue().splitlines()
    print(json.dumps({
        "lens": lens.tolist(), "span_lens": rows.span[:, 1].tolist(),
        "strings": [None if s is None else s.decode() for s in strings],
        "f": [None if v != v else v for v in rows.f64[:, 0].tolist()],
        "gather_addr": hex(GATHER_ADDR),
        "saved": [l.rsplit("/", 1)[-1] for l in lines if "[cache] data saved" in l],
        "loaded": [l.rsplit("/", 1)[-1] for l in lines if "[cache] data loaded" in l],
    }))
    """
)


def _run_child(code: str, tmp_path: Path, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    src_dir = str(Path(decider2.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src_dir, env.get("PYTHONPATH")) if p)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "numba_cache")
    env["NUMBA_DEBUG_CACHE"] = "1"
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=600, check=False,
    )


def test_a_kernel_calling_the_shim_through_a_pointer_argument_is_a_genuine_cache_hit(tmp_path):
    def run():
        proc = _run_child(_CACHE_CHILD, tmp_path)
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.splitlines()[-1])

    cold, warm = run(), run()
    for r in (cold, warm):
        assert r["lens"] == r["span_lens"] == [3, -1, 33]
        assert r["strings"] == ["dog", None, "a much longer merchant descriptor"]
        assert r["f"] == [1.0, None, 3.0]
    ours = ("materialize_rows", "string_lengths")
    cold_saved = [n for n in cold["saved"] if any(k in n for k in ours)]
    warm_loaded = [n for n in warm["loaded"] if any(k in n for k in ours)]
    assert len(cold_saved) >= 2 and not [n for n in cold["loaded"] if any(k in n for k in ours)], cold
    assert len(warm_loaded) >= 2 and not [n for n in warm["saved"] if any(k in n for k in ours)], warm


# ---------------------------------------------------------------------------
# the artefact: what loaded, what it exports, where it came from
# ---------------------------------------------------------------------------

def test_diagnose_reports_the_loaded_shim_and_the_vendor_pin():
    d = arrow.diagnose()
    assert arrow.available()
    assert d["shim"] == "loaded" and d["nanoarrow"] == "0.9.0" and d["abi"] == 1
    assert d["vendor_pin"].startswith("nanoarrow 0.9.0 (apache-arrow-nanoarrow-0.9.0.tar.gz")
    assert d["vendor_manifest"] == "SHA512SUMS present"
    assert Path(d["extension"]).name.startswith("_nashim")
    assert d["python"] == platform.python_version() and d["platform"] == platform.platform()
    assert d["sizes"]["coldesc"] == 64 and d["sizes"]["plan"] == 64


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("nm") is None, reason="nm -D is an ELF check")
def test_the_extension_exports_exactly_one_symbol():
    so = arrow.diagnose()["extension"]
    out = subprocess.run(["nm", "-D", "--defined-only", so], capture_output=True, text=True, check=True).stdout
    symbols = [line.split()[-1] for line in out.splitlines() if line.strip()]
    assert symbols == ["PyInit__nashim"], symbols


def test_the_vendored_nanoarrow_matches_its_manifest_and_the_header_matches_the_pin():
    vendor = Path(decider2.__file__).parent / "_arrow" / "vendor"
    if not (vendor / "nanoarrow.c").is_file():
        pytest.skip("C sources are not shipped in the wheel; this is an sdist/editable check")
    for line in (vendor / "SHA512SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha512((vendor / name).read_bytes()).hexdigest() == digest, name
    pin = (vendor / "VERSION").read_text().split()[1]
    header = (vendor / "nanoarrow" / "nanoarrow.h").read_text()
    assert f'#define NANOARROW_VERSION "{pin}"' in header
    assert arrow.diagnose()["nanoarrow"] == pin


_MISSING_CHILD = textwrap.dedent(
    """
    import json, sys
    sys.modules["decider2._arrow._nashim"] = None      # what the import system sees with no extension
    import decider2._arrow as a
    d = a.diagnose()
    assert d["shim"] == "unavailable", d
    try:
        from decider2._arrow import FrameView
    except ImportError as e:
        print(json.dumps({"error": str(e), "diagnose_error": d["error"]}))
    else:
        print(json.dumps({"error": None}))
    """
)


def test_a_missing_extension_is_one_clear_import_error_naming_platform_interpreter_and_cause(tmp_path):
    proc = _run_child(_MISSING_CHILD, tmp_path)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout.splitlines()[-1])
    msg = got["error"]
    assert msg is not None
    assert "decider2._arrow._nashim" in msg and "could not be loaded" in msg
    assert platform.platform() in msg and platform.python_version() in msg and platform.machine() in msg
    assert "ModuleNotFoundError:" in msg and "halted" in msg            # the underlying cause is embedded
    assert "no pure-Python path" in msg and "wheel" in msg
    assert got["diagnose_error"] == msg                               # diagnose() reports the same text


_TRUNCATED_CHILD = textwrap.dedent(
    """
    import json, decider2
    try:
        import decider2._arrow.frame
    except ImportError as e:
        print(json.dumps({"file": decider2.__file__, "error": str(e)}))
    else:
        print(json.dumps({"file": decider2.__file__, "error": None}))
    """
)


def test_a_truncated_extension_file_reports_the_loader_error_with_platform_and_interpreter(tmp_path):
    src = Path(decider2.__file__).parent
    site = tmp_path / "site"
    shutil.copytree(src, site / "decider2", ignore=shutil.ignore_patterns("__pycache__", "*.c", "*.h"))
    sos = list((site / "decider2" / "_arrow").glob("_nashim*"))
    assert len(sos) == 1, sos
    sos[0].write_bytes(b"not an extension module")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(site)
    proc = subprocess.run([sys.executable, "-c", _TRUNCATED_CHILD], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout.splitlines()[-1])
    assert got["file"].startswith(str(site)), got                    # the corrupted copy was the one imported
    msg = got["error"]
    assert msg is not None and "could not be loaded" in msg
    assert platform.platform() in msg and platform.python_version() in msg
    assert "ImportError:" in msg and "_nashim" in msg                 # the dynamic loader's own words follow
