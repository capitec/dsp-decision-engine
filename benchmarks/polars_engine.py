"""The polars host against eager fused and hand-written polars, on three pipelines.

Pipelines: `flagship` (arithmetic), `fallback` (ten str/date steps that run as Python fallbacks) and `tree` (127
nodes). Configurations: `fused-mem`, `fused-io` (eager, the io one reads and writes parquet), `host-mem`,
`host-memstream` (the frame in memory, collected by the streaming engine), `host-stream` (scan to sink_parquet),
`native-mem`, `native-stream` (hand-written polars) and, for `fallback`,
`frames-*` (ten frame steps with polars bodies) and `frame1-*` (one frame step); `flagship` also has `copy-stream`, a
plain scan to sink. Every run is a child process. `E2_ROWGROUP` sets the input parquet row-group size (a separate
data folder), `E2_CHUNK` the streaming chunk size.

    uv run python benchmarks/polars_engine.py check                       # outputs equal fused, 10k rows
    uv run python benchmarks/polars_engine.py run --sizes 10k 1m --out r.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import statistics
import subprocess
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from decikit import flow, frame_step, param

sys.path.insert(0, str(Path(__file__).parent))
import tree_walk  # noqa: E402  (its generator and document; its main block does not run)

from decikit.steps.trees import TreeConfig  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / ".scratch" / "e2-data"
SIZES = {"10k": 10_000, "1m": 1_000_000, "5m": 5_000_000, "50m": 50_000_000}
CHUNK = 2_500_000
# Input row-group size and streaming chunk size are set from the environment so a run can vary them.
ROWGROUP = int(os.environ.get("E2_ROWGROUP", 1_000_000))
MEMORY_MAX = "7G"


# ---- flagship -------------------------------------------------------------------------------------------------
def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def cap_by_income_band(term_cap: float, min_net_salary: float, cap: float = param(48.0, ge=6, le=60),
                       income_threshold: float = param(5000.0, ge=0)) -> float:
    if min_net_salary < income_threshold:
        return min(term_cap, cap)
    return term_cap


def flagship_native(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        affordability_ratio=(pl.col("net_income") - pl.col("expenses")) / pl.col("instalment"),
        cap_by_income_band=pl.when(pl.col("min_net_salary") < 5000.0)
        .then(pl.min_horizontal("term_cap", pl.lit(48.0))).otherwise(pl.col("term_cap")))


def flagship_frame(n: int, rng: np.random.Generator) -> pl.DataFrame:
    return pl.DataFrame({
        "net_income": rng.uniform(3000, 20000, n), "expenses": rng.uniform(500, 3000, n),
        "instalment": rng.uniform(200, 3000, n), "term_cap": np.full(n, 60.0),
        "min_net_salary": rng.uniform(3000, 20000, n)})


# ---- fallback-heavy: ten str/date steps, each a plain Python body ------------------------------------------------
REGIONS = {"WC": "Western Cape", "GP": "Gauteng", "KZ": "KwaZulu-Natal", "EC": "Eastern Cape", "LP": "Limpopo",
           "FS": "Free State", "NW": "North West", "MP": "Mpumalanga", "NC": "Northern Cape"}


def tenure_months(open_date: date, as_of: date) -> int:
    return (as_of.year - open_date.year) * 12 + as_of.month - open_date.month - (as_of.day < open_date.day)


def age_years(birth_date: date, as_of: date) -> int:
    return as_of.year - birth_date.year - ((as_of.month, as_of.day) < (birth_date.month, birth_date.day))


def opened_on_weekend(open_date: date) -> bool:
    return open_date.weekday() >= 5


def is_month_end(as_of: date) -> bool:
    return (as_of + timedelta(days=1)).day == 1


def clean_name(name: str) -> str:
    return " ".join(name.split()).upper()


def product_prefix(account_code: str) -> str:
    return account_code.split("-")[0]


def is_closed_account(account_code: str) -> bool:
    return account_code.endswith("-X")


def region(branch_code: str) -> str:
    return REGIONS.get(branch_code[:2], "Other")


def branch_number(branch_code: str) -> int:
    return int(branch_code.split("-")[1])


def segment(region: str, tenure_months: int) -> str:
    return f"{region}:{'new' if tenure_months < 12 else 'established'}"


def _ymd(c: str) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    return tuple(getattr(pl.col(c).dt, p)().cast(pl.Int64) for p in ("year", "month", "day"))


def _fallback_exprs() -> dict[str, tuple[list[str], pl.DataType, pl.Expr]]:
    """name -> (columns read, dtype, polars expression), in dependency order."""
    (oy, om, od), (ay, am, ad), (by, bm, bd) = _ymd("open_date"), _ymd("as_of"), _ymd("birth_date")
    return {
        "tenure_months": (["open_date", "as_of"], pl.Int64, (ay - oy) * 12 + am - om - (ad < od).cast(pl.Int64)),
        "age_years": (["birth_date", "as_of"], pl.Int64,
                      ay - by - ((am < bm) | ((am == bm) & (ad < bd))).cast(pl.Int64)),
        "opened_on_weekend": (["open_date"], pl.Boolean, pl.col("open_date").dt.weekday() >= 6),
        "is_month_end": (["as_of"], pl.Boolean, pl.col("as_of").dt.month_end() == pl.col("as_of")),
        "clean_name": (["name"], pl.String,
                       pl.col("name").str.replace_all(r"\s+", " ").str.strip_chars().str.to_uppercase()),
        "product_prefix": (["account_code"], pl.String, pl.col("account_code").str.split("-").list.get(0)),
        "is_closed_account": (["account_code"], pl.Boolean, pl.col("account_code").str.ends_with("-X")),
        "region": (["branch_code"], pl.String,
                   pl.col("branch_code").str.slice(0, 2).replace_strict(REGIONS, default="Other")),
        "branch_number": (["branch_code"], pl.Int64,
                          pl.col("branch_code").str.split("-").list.get(1).cast(pl.Int64)),
        "segment": (["region", "tenure_months"], pl.String,
                    pl.concat_str(pl.col("region"), pl.lit(":"),
                                  pl.when(pl.col("tenure_months") < 12).then(pl.lit("new"))
                                  .otherwise(pl.lit("established")))),
    }


FALLBACK_CONSUMED = ["region", "tenure_months"]


def fallback_native(lf: pl.LazyFrame) -> pl.LazyFrame:
    exprs = _fallback_exprs()
    # Two rounds: `segment` reads two columns the first round makes.
    first = {k: v[2] for k, v in exprs.items() if k != "segment"}
    return lf.with_columns(**first).with_columns(segment=exprs["segment"][2]).drop(FALLBACK_CONSUMED)


def fallback_frames() -> list:
    def make(name, reads, dtype, expr):
        def body(df):
            return df.with_columns(expr.alias(name))
        body.__name__ = name
        return frame_step(body, reads=reads, writes={name: dtype})

    return [make(name, *spec) for name, spec in _fallback_exprs().items()]


def fallback_one_frame():
    def fallback_columns(df: pl.DataFrame) -> pl.DataFrame:
        return fallback_native(df.lazy()).collect()
    return frame_step(fallback_columns, reads=["open_date", "as_of", "birth_date", "name", "account_code", "branch_code"],
                      writes={k: v[1] for k, v in _fallback_exprs().items() if k not in FALLBACK_CONSUMED})


NAMES = ["  thandi  Mokoena", "JOHN  smith ", "Sipho Dlamini", " nomsa van der merwe", "Aisha  KHAN", "peter  Botha  ",
         "Lerato Mahlangu", "  zanele   NKOSI", "David O Brien", "Fatima  Patel ", "mpho  Sithole", "Anele   Ndlovu"]


def fallback_frame(n: int, rng: np.random.Generator) -> pl.DataFrame:
    opened = rng.integers(14610, 20000, n)  # 2010-01-01 onwards
    df = pl.DataFrame({
        "open_date": opened.astype(np.int32), "as_of": (opened + rng.integers(0, 1500, n)).astype(np.int32),
        "birth_date": rng.integers(-7000, 12000, n).astype(np.int32),
        "name": pl.Series(NAMES).gather(rng.integers(0, len(NAMES), n)),
        "_p": pl.Series(["RET", "CRD", "LON", "OVD"]).gather(rng.integers(0, 4, n)),
        "_n": rng.integers(0, 99999, n), "_s": pl.Series(["A", "B", "X"]).gather(rng.integers(0, 3, n)),
        "_r": pl.Series([*REGIONS, "XX"]).gather(rng.integers(0, 10, n)), "_b": rng.integers(1000, 9999, n)})
    return df.select(
        pl.col("open_date", "as_of", "birth_date").cast(pl.Date), "name",
        account_code=pl.concat_str("_p", pl.lit("ZA"), pl.col("_n").cast(pl.String).str.zfill(5), "_s", separator="-"),
        branch_code=pl.concat_str("_r", pl.col("_b").cast(pl.String), separator="-"))


# ---- tree: the 127-node document from tree_walk.py ---------------------------------------------------------------
def _tree_leaf_expr(doc: dict) -> pl.Expr:
    nodes = {n["id"]: n["data"] for n in doc["nodes"]}
    kids: dict[str, dict[int, str]] = {}
    for e in doc["edges"]:
        kids.setdefault(e["source"], {})[e["data"]["sourceIndex"][0]] = e["target"]

    def cond(c: dict) -> pl.Expr:
        col = pl.col(c["feature"])
        if c["op"] == "between":
            return col.is_between(c["min"], c["max"])
        return col >= c["threshold"] if c["op"] == ">=" else col < c["threshold"]

    def expr(nid: str) -> pl.Expr:
        d = nodes[nid]
        if d["type"] == "leaf":
            return pl.lit(d["result_idx"], dtype=pl.Int64)
        test = cond(d["condition"]) if d["type"] == "unary" else pl.all_horizontal(cond(c) for c in d["conditions"])
        return pl.when(test).then(expr(kids[nid][0])).otherwise(expr(kids[nid][1]))

    return expr(doc["nodes"][0]["id"])


def tree_native(lf: pl.LazyFrame) -> pl.LazyFrame:
    # One when-chain finds the leaf; each output column is a lookup on it (a translator would do no better).
    rows = tree_walk.DOC["output"]["data"]
    ids = list(range(len(rows)))
    return lf.with_columns(_leaf=_tree_leaf_expr(tree_walk.DOC)).with_columns(
        **{c: pl.col("_leaf").replace_strict(ids, [r[c] for r in rows], return_dtype=t)
           for c, t in (("label", pl.String), ("score", pl.Float64), ("band", pl.Int64))}).drop("_leaf")


def tree_frame(n: int, rng: np.random.Generator) -> pl.DataFrame:
    return pl.DataFrame({**{f: rng.uniform(0, 1, n) for f in tree_walk.FLOATS},
                         **{f: rng.integers(0, 24, n) for f in tree_walk.INTS}})


# ---- pipelines and configurations ----------------------------------------------------------------------------------
PIPELINES = {
    "flagship": dict(make=lambda: flow(disposable_income, affordability_ratio, cap_by_income_band),
                     native=flagship_native, gen=flagship_frame),
    "fallback": dict(make=lambda: flow(tenure_months, age_years, opened_on_weekend, is_month_end, clean_name,
                                       product_prefix, is_closed_account, region, branch_number, segment),
                     native=fallback_native, gen=fallback_frame,
                     frames=lambda: flow(*fallback_frames()), frame1=lambda: flow(fallback_one_frame())),
    "tree": dict(make=lambda: TreeConfig(name="campaign", tree=tree_walk.DOC, feature_types=tree_walk.TYPES),
                 native=tree_native, gen=tree_frame),
}


def configs(pipe: str) -> list[str]:
    engines = ["fused", "host", "native"] + [e for e in ("frames", "frame1") if e in PIPELINES[pipe]]
    out = []
    for e in engines:
        if e == "fused":
            out += ["fused-mem", "fused-io"]
        else:
            # host-memstream: the in-memory frame through the host, collected by the streaming engine.
            out += [f"{e}-mem"] + (["host-memstream"] if e == "host" else []) + [f"{e}-stream"]
    # A scan straight into a sink: what streaming costs in memory before any step runs.
    return out + (["copy-stream"] if pipe == "flagship" else [])


def bind(step, engine: str):
    """Fused or host executable of a step, by feature: today's API or whatever replaces it."""
    kind = "fused" if engine == "fused" else "polars"
    try:
        return step.compile(kind)
    except Exception:
        if kind == "fused":
            from decikit._engine import Engine
            return Engine().bind(step, mode="fused")
        from decikit._engine.polars.engine import PolarsEngine
        return PolarsEngine().bind(step)


def runner(pipe: str, engine: str):
    """A function from a LazyFrame (a DataFrame for fused) to the pipeline's output, plus the build seconds."""
    p = PIPELINES[pipe]
    t = time.perf_counter()
    if engine in ("native", "copy"):
        return (p["native"] if engine == "native" else lambda lf: lf), 0.0
    exe = bind(p["make" if engine in ("fused", "host") else engine](), "fused" if engine == "fused" else "host")
    return exe.run, time.perf_counter() - t


# ---- data ------------------------------------------------------------------------------------------------------------
def data_path(pipe: str, size: str) -> str:
    # A size above one chunk is a folder of parquet parts read as a glob: merging them needs more memory than the cap.
    tag = "" if ROWGROUP == 1_000_000 else f"-rg{ROWGROUP}"
    return (str(DATA / f"{pipe}-{size}{tag}" / "*.parquet") if SIZES[size] > CHUNK
            else str(DATA / f"{pipe}-{size}{tag}.parquet"))


def ensure_data(pipe: str, size: str) -> str:
    path = data_path(pipe, size)
    if glob.glob(path):
        return path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = SIZES[size]
    for i, start in enumerate(range(0, n, CHUNK)):
        part = path.replace("*", f"part{i:03d}") if n > CHUNK else path
        PIPELINES[pipe]["gen"](min(CHUNK, n - start), np.random.default_rng([i, 7])).write_parquet(
            part, row_group_size=ROWGROUP)
    return path


# ---- one child run -----------------------------------------------------------------------------------------------------
def child(pipe: str, cfg: str, size: str) -> dict:
    engine, mode = cfg.split("-")
    path = ensure_data(pipe, size)
    n = SIZES[size]
    if os.environ.get("E2_CHUNK"):
        pl.Config.set_streaming_chunk_size(int(os.environ["E2_CHUNK"]))
    run, build = runner(pipe, engine)
    calls = []  # one entry per batch the host hands a segment
    try:
        from decikit._engine.polars import host
        original = host._run_batch
        host._run_batch = lambda *a: (calls.append(1), original(*a))[1]
    except ImportError:
        pass
    t = time.perf_counter()
    warm = pl.read_parquet(path, n_rows=1000)
    (run(warm) if engine == "fused" else run(warm.lazy()).collect())
    build += time.perf_counter() - t
    out = DATA / f"out-{os.getpid()}.parquet"
    load = 0.0
    if mode == "stream":
        def once():
            run(pl.scan_parquet(path)).sink_parquet(out)
    elif mode == "io":
        def once():
            run(pl.read_parquet(path)).write_parquet(out)
    else:
        t = time.perf_counter()
        df = pl.read_parquet(path)
        load = time.perf_counter() - t
        lf = df.lazy()

        def once():
            if engine == "fused":
                run(df)
            else:
                run(lf).collect(engine="streaming") if mode == "memstream" else run(lf).collect()
    # A fast run repeats in the child (up to 20 times or 3 s) and reports the median; a slow one runs once.
    walls = []
    while not walls or (len(walls) < 20 and sum(walls) < 3):
        before = len(calls)
        t = time.perf_counter()
        once()
        walls.append(time.perf_counter() - t)
        batches = len(calls) - before
    out.unlink(missing_ok=True)
    wall = statistics.median(walls)
    return {"pipeline": pipe, "config": cfg, "size": size, "rows": n, "status": "ok", "build_s": round(build, 3),
            "load_s": round(load, 3), "wall_s": round(wall, 4), "first_s": round(walls[0], 4), "inner": len(walls), "batches": batches,
            "threads": pl.thread_pool_size(), "rowgroup": ROWGROUP, "chunk": os.environ.get("E2_CHUNK"), "rows_per_s": round(n / wall),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}


# ---- driver --------------------------------------------------------------------------------------------------------------
def spawn(pipe: str, cfg: str, size: str, rep: int, timeout: float) -> dict:
    base = {"pipeline": pipe, "config": cfg, "size": size, "rows": SIZES[size], "rep": rep}
    t = time.perf_counter()
    try:
        # Its own memory-capped scope, so an out-of-memory kill takes the child and not the driver.
        r = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={MEMORY_MAX}",
                            "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue",
                            sys.executable, __file__, "child", pipe, cfg, size],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {**base, "status": "timeout", "wall_s": round(time.perf_counter() - t)}
    if r.returncode == 0:
        return {**base, **json.loads(r.stdout.strip().splitlines()[-1]), "rep": rep}
    status = "oom" if r.returncode in (-9, 137) else "error"
    return {**base, "status": status, "returncode": r.returncode, "stderr": r.stderr[-600:],
            "wall_s": round(time.perf_counter() - t)}


def meta(tag: str) -> dict:
    return {"meta": tag, "threads": pl.thread_pool_size(), "load_avg": os.getloadavg(), "polars": pl.__version__,
            "time": time.strftime("%H:%M:%S")}


def drive(a: argparse.Namespace) -> None:
    with open(a.out, "a") as f:
        def emit(row: dict) -> None:
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row), flush=True)

        emit(meta("start"))
        for pipe in a.pipelines:
            for size in a.sizes:
                ensure_data(pipe, size)
                cfgs = [c for c in configs(pipe) if not a.configs or c in a.configs]
                for rep in range(a.reps):
                    for cfg in cfgs:
                        emit(spawn(pipe, cfg, size, rep, a.timeout))
        emit(meta("end"))


# ---- correctness -----------------------------------------------------------------------------------------------------------
def differences(ref: pl.DataFrame, got: pl.DataFrame) -> list[str]:
    if sorted(ref.columns) != sorted(got.columns):
        return [f"columns differ: missing {sorted(set(ref.columns) - set(got.columns))}, "
                f"extra {sorted(set(got.columns) - set(ref.columns))}"]
    if ref.height != got.height:
        return [f"{got.height} rows against {ref.height}"]
    found = []
    for c in sorted(ref.columns):
        a, b = ref[c], got[c]
        if a.dtype != b.dtype:
            found.append(f"{c}: dtype {b.dtype} against {a.dtype}")
        elif not a.equals(b):
            bad = (a != b).fill_null(False) | (a.is_null() != b.is_null())
            msg = f"{c}: {int(bad.sum())} of {a.len()} rows differ"
            if a.dtype.is_float():
                msg += f", max abs diff {(a - b).abs().max()}"
            found.append(msg)
    return found


def check(a: argparse.Namespace) -> None:
    bad = 0
    for pipe in a.pipelines:
        path = ensure_data(pipe, "10k")
        df = pl.read_parquet(path)
        run, _ = runner(pipe, "fused")
        ref = run(df)
        for cfg in configs(pipe):
            engine, mode = cfg.split("-")
            if engine in ("fused", "copy"):
                continue
            run, _ = runner(pipe, engine)
            if mode in ("mem", "memstream"):
                pl.Config.set_streaming_chunk_size(1000)
                got = run(df.lazy()).collect(engine="streaming" if mode == "memstream" else "auto")
                pl.Config.restore_defaults()
            else:
                out = DATA / f"check-{pipe}-{cfg}.parquet"
                pl.Config.set_streaming_chunk_size(1000)
                run(pl.scan_parquet(path)).sink_parquet(out)
                pl.Config.restore_defaults()
                got = pl.read_parquet(out)
                out.unlink()
            diff = differences(ref, got)
            bad += bool(diff)
            print(f"{pipe:9} {cfg:14} {'equal' if not diff else 'DIFFERENT'}", *diff, sep="\n    " if diff else "")
    sys.exit(1 if bad else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("child")
    c.add_argument("pipeline"), c.add_argument("config"), c.add_argument("size")
    for name in ("run", "check"):
        s = sub.add_parser(name)
        s.add_argument("--pipelines", nargs="+", default=list(PIPELINES), choices=list(PIPELINES))
        if name == "run":
            s.add_argument("--sizes", nargs="+", default=["10k", "1m"], choices=list(SIZES))
            s.add_argument("--configs", nargs="*", help="only these configurations")
            s.add_argument("--reps", type=int, default=3)
            s.add_argument("--timeout", type=float, default=3600)
            s.add_argument("--out", default="e2-results.jsonl")
    a = ap.parse_args()
    warnings.simplefilter("ignore")  # every fallback step warns, and the hidden-state warning names REGIONS
    if a.cmd == "child":
        print(json.dumps(child(a.pipeline, a.config, a.size)))
    else:
        {"run": drive, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
