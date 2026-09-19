"""
EXPERIMENT J2 -- shared shape/codegen/estimator module for the chunked
write-back experiment (doc 01 S4d / doc 05 S3, continuing experiment J in
experimentation/output-writeback-convention/).

Same 400-input / 633-output shape as experiment J, held identical so J2's
numbers are directly comparable to results_100k.json:

  IN:  300 f8 + 80 i8 + 20 bool  = 400
  OUT: 380 f8 + 158 i8 + 95 bool = 633

Reused verbatim from experimentation/output-writeback-convention/writeback.py
(experiment J): the shape constants, the header/body codegen for the
record-out and dtype-grouped-2D-out conventions, extract() via
Series._get_buffers() (no pyarrow, per EXPERIMENTS.md SA), and the
zero-copy check via _get_buffer_info(). NOT reused as-is: experiment J used
cache=False and a per-call UID'd generated file (fresh file every call, so
every subprocess recompiled). J2 instead writes the generated driver ONCE
to a FIXED path and sets cache=True, so repeat subprocess invocations across
this sweep get a numba on-disk cache HIT instead of paying compile again --
this is the "content-addressed driver naming" mitigation EXPERIMENTS.md SC
recommended, applied for real, using SC's own six-condition cache contract
(same path, same bytes, same mtime, same signature, same magic_tuple).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
GEN_DIR = HERE / "_generated"
GEN_DIR.mkdir(exist_ok=True)
if str(GEN_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_DIR))

IN_F8, IN_I8, IN_BOOL = 300, 80, 20
N_IN = IN_F8 + IN_I8 + IN_BOOL
assert N_IN == 400

OUT_F8, OUT_I8, OUT_BOOL = 380, 158, 95
N_OUT = OUT_F8 + OUT_I8 + OUT_BOOL
assert N_OUT == 633

IN_F8_NAMES = [f"i_f8_{k}" for k in range(IN_F8)]
IN_I8_NAMES = [f"i_i8_{k}" for k in range(IN_I8)]
IN_B_NAMES = [f"i_b_{k}" for k in range(IN_BOOL)]
IN_ALL_NAMES = IN_F8_NAMES + IN_I8_NAMES + IN_B_NAMES

OUT_F8_NAMES = [f"o_f8_{k}" for k in range(OUT_F8)]
OUT_I8_NAMES = [f"o_i8_{k}" for k in range(OUT_I8)]
OUT_B_NAMES = [f"o_b_{k}" for k in range(OUT_BOOL)]
OUT_ALL_NAMES = OUT_F8_NAMES + OUT_I8_NAMES + OUT_B_NAMES

IN_DTYPE = np.dtype(
    [(n, "f8") for n in IN_F8_NAMES]
    + [(n, "i8") for n in IN_I8_NAMES]
    + [(n, "?") for n in IN_B_NAMES]
)
OUT_REC_DTYPE = np.dtype(
    [(n, "f8") for n in OUT_F8_NAMES]
    + [(n, "i8") for n in OUT_I8_NAMES]
    + [(n, "?") for n in OUT_B_NAMES]
)

# bytes/row, used for the pre-run estimate (MEASURE rule 1: print the
# estimate before running).
BYTES_PER_ROW_IN = IN_F8 * 8 + IN_I8 * 8 + IN_BOOL * 1     # 3060
BYTES_PER_ROW_OUT = OUT_F8 * 8 + OUT_I8 * 8 + OUT_BOOL * 1  # 4399

_HEADER = """\
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def driver(inp, coef_f8, {out_args}):
    n = inp.shape[0]
    for i in range(n):
        a0 = inp[i].i_f8_0
        a1 = inp[i].i_f8_1
        a2 = inp[i].i_f8_2
        a3 = inp[i].i_f8_3
        a4 = inp[i].i_f8_4
        b0 = inp[i].i_i8_0
"""


def gen_record_out_source() -> str:
    lines = [_HEADER.format(out_args="out")]
    for j, name in enumerate(OUT_F8_NAMES):
        lines.append(f"        out[i].{name} = coef_f8[{j}] * a0 + a1 - a2 * 0.001")
    for j, name in enumerate(OUT_I8_NAMES):
        lines.append(f"        out[i].{name} = b0 + {j}")
    for j, name in enumerate(OUT_B_NAMES):
        lines.append(f"        out[i].{name} = (a3 + {j} * 0.0001) > a4")
    return "\n".join(lines) + "\n"


def gen_2d_source(row_major: bool) -> str:
    lines = [_HEADER.format(out_args="out_f8, out_i8, out_b")]
    if row_major:
        f8_lhs, i8_lhs, b_lhs = "out_f8[i, j]", "out_i8[i, j]", "out_b[i, j]"
    else:
        f8_lhs, i8_lhs, b_lhs = "out_f8[j, i]", "out_i8[j, i]", "out_b[j, i]"
    lines.append(f"        for j in range({OUT_F8}):")
    lines.append(f"            {f8_lhs} = coef_f8[j] * a0 + a1 - a2 * 0.001")
    lines.append(f"        for j in range({OUT_I8}):")
    lines.append(f"            {i8_lhs} = b0 + j")
    lines.append(f"        for j in range({OUT_BOOL}):")
    lines.append(f"            {b_lhs} = (a3 + j * 0.0001) > a4")
    return "\n".join(lines) + "\n"


def fixed_driver_path(variant: str) -> Path:
    return GEN_DIR / f"driver_{variant}.py"


def load_driver(variant: str):
    """Write the generated driver to a FIXED path, only if not already
    present with the exact same bytes (never rewrite unchanged content --
    experiment C showed a byte-identical rewrite at the same path still
    changes mtime and MISSES; only a truly untouched file HITs). Then import
    it via spec_from_file_location (real file, never exec()), per doc 05
    S4.1, same as experiment J."""
    if variant == "record":
        src = gen_record_out_source()
    elif variant == "col_major":
        src = gen_2d_source(row_major=False)
    elif variant == "row_major":
        src = gen_2d_source(row_major=True)
    else:
        raise ValueError(variant)

    path = fixed_driver_path(variant)
    if not path.exists() or path.read_text() != src:
        path.write_text(src)

    # Import by MODULE NAME (GEN_DIR is on sys.path) rather than
    # spec_from_file_location, so the module is a normal, importable-by-name
    # entry. This matters for cache survival across processes: numba's
    # on-disk cache pickles a reference to the function's environment module
    # and rebuilds it with importlib.import_module(modname) in whatever
    # process loads the cache -- spec_from_file_location never registers
    # that name anywhere importlib.import_module can find it, so a fresh
    # process hits `ModuleNotFoundError: No module named '<dynamic>'` even
    # with byte-identical file + mtime (see README.md "cache note").
    modname = f"driver_{variant}"
    importlib.invalidate_caches()
    if modname in sys.modules:
        del sys.modules[modname]
    mod = importlib.import_module(modname)
    return mod.driver, path


def make_input_frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {}
    for name in IN_F8_NAMES:
        cols[name] = rng.random(n)
    for name in IN_I8_NAMES:
        cols[name] = rng.integers(-1000, 1000, n, dtype=np.int64)
    for name in IN_B_NAMES:
        cols[name] = rng.random(n) < 0.5
    return pl.DataFrame(cols)


def extract(frame: pl.DataFrame) -> dict:
    """Per-column, via polars-native buffers -- no pyarrow (EXPERIMENTS.md SA)."""
    return {name: frame[name]._get_buffers()["values"].to_numpy() for name in IN_ALL_NAMES}


def assemble(cols: dict, n: int) -> np.ndarray:
    rec = np.empty(n, dtype=IN_DTYPE)
    for name in IN_ALL_NAMES:
        rec[name] = cols[name]
    return rec


def writeback_record(out_rec: np.ndarray) -> pl.DataFrame:
    d = {name: out_rec[name] for name in OUT_ALL_NAMES}
    return pl.DataFrame(d)


def writeback_2d(out_f8: np.ndarray, out_i8: np.ndarray, out_b: np.ndarray, row_major: bool) -> pl.DataFrame:
    d = {}
    if row_major:
        for j, name in enumerate(OUT_F8_NAMES):
            d[name] = out_f8[:, j]
        for j, name in enumerate(OUT_I8_NAMES):
            d[name] = out_i8[:, j]
        for j, name in enumerate(OUT_B_NAMES):
            d[name] = out_b[:, j]
    else:
        for j, name in enumerate(OUT_F8_NAMES):
            d[name] = out_f8[j, :]
        for j, name in enumerate(OUT_I8_NAMES):
            d[name] = out_i8[j, :]
        for j, name in enumerate(OUT_B_NAMES):
            d[name] = out_b[j, :]
    return pl.DataFrame(d)


def estimate_peak_bytes(variant: str, n: int) -> int:
    """Analytic pre-run estimate (MEASURE rule 1). Model: in_rec + out
    array(s) resident together, PLUS a full duplicate of the output for the
    'record' variant only (writeback_record() extracts named fields out of a
    structured array, which are strided views -> pl.DataFrame(dict) copies
    them; col_major's writeback slices contiguous rows of a (OUT_F8, n)
    array -> zero-copy, confirmed by experiment J's zero_copy_sample, so no
    duplicate). Excludes the ~150-250MB python+numba+polars baseline RSS
    that isn't proportional to n -- the measured numbers below include it.
    """
    base = n * (BYTES_PER_ROW_IN + BYTES_PER_ROW_OUT)
    if variant == "record":
        base += n * BYTES_PER_ROW_OUT
    return base
