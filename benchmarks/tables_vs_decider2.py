"""Decision-table score() latency and batch throughput: decider's DecisionTableConfig against decider2's table_module.

Two tables. "bands": 12 contiguous score bands (between) writing a String,
an Int64 and a Float64 column. "gated": the same bands ANDed with a string
`in` on a `channel` column (a List[String] per row). decider2 writes String
columns with `decode()` after the run, which is included in its batch time.

decider2's compiled path needs its `_nashim` extension built for this interpreter:

    PYTHONPATH=<decider2 src with _nashim built> uv run python benchmarks/tables_vs_decider2.py
"""
import gc
import time

import numpy as np
import polars as pl

from decider.engine import Engine
from decider.steps.tables import DecisionTableConfig
from decider2 import flow as flow2
from decider2.tables import DecisionTable, table_module

N = 1_000_000
rng = np.random.default_rng(0)
CUTS = np.linspace(0.0, 1.0, 13).tolist()
ROWS = [{"lo": None if i == 0 else CUTS[i], "hi": None if i == 11 else CUTS[i + 1], "band": f"band_{i % 5}",
         "pts": i, "rate": i / 10, "keys": ["app", "web"] if i % 2 else ["branch"]} for i in range(12)]
DTYPES = {"lo": "Float64", "hi": "Float64", "band": "String", "pts": "Int64", "rate": "Float64",
          "keys": {"type": "List", "inner": "String"}}
BETWEEN = {"type": "between", "variable": "score", "lower_bound_column": "lo", "upper_bound_column": "hi"}
GATED = {"type": "and", "expressions": [BETWEEN, {"type": "in", "variable": "channel", "values_column": "keys"}]}
OUTPUTS = ["band", "pts", "rate"]
frame = pl.DataFrame({"score": rng.uniform(0, 1, N), "channel": rng.choice(["app", "web", "branch", "call"], N)})
ROW = frame.row(0, named=True)


def _time(f, *args):
    t = time.perf_counter()
    f(*args)
    return time.perf_counter() - t


def rows_per_s(run, reps=7):
    run(frame)
    return N / min(_time(run, frame) for _ in range(reps))


def latency(score, calls=20000):
    for _ in range(1000):
        score(ROW)
    gc.collect()
    samples = sorted(_time(score, ROW) for _ in range(calls))
    return samples[len(samples) // 2] * 1e6, samples[int(len(samples) * 0.99)] * 1e6


def compare(expression: dict) -> None:
    doc = {"parameters": {"data": ROWS, "dtypes": list(DTYPES.items())}, "expression": expression,
           "outputs": OUTPUTS, "default": ["none", -1, 0.0]}
    built = table_module(DecisionTable.model_validate({"name": "bands", **doc}))
    pipeline2 = flow2(built.module)
    pipeline2.precompile(shared=built.shared)
    engines = {"decider2 fused": (lambda df: built.decode(pipeline2.apply(df, shared=built.shared, mode="fused")),
                                  lambda r: pipeline2.score(r, shared=built.shared))}
    table = DecisionTableConfig.load({"type": "decision_table", "name": "bands", **doc})
    for mode in ("fused", "stepped", "interpreted"):
        exe = Engine().bind(table, mode=mode)
        engines[f"decider {mode}"] = (exe.run, exe.score)
    # The same rows read from the params document, which every call passes.
    columns = {**DTYPES, "band": {"type": "Enum", "categories": sorted({r["band"] for r in ROWS} | {"none"})}}
    by_param = DecisionTableConfig.load({"type": "decision_table", "name": "bands", "columns": columns,
                                         "rows": {"table": "rows"},
                                         **{k: v for k, v in doc.items() if k != "parameters"}})
    exe = Engine().bind(by_param, mode="fused")
    params = {"bands": {"rows": ROWS}}
    engines["decider fused, rows param"] = (lambda df, exe=exe: exe.run(df, params),
                                            lambda r, exe=exe: exe.score(r, params))

    # The outputs agree before anything is timed.
    expected = engines["decider2 fused"][0](frame.head(20000)).select(OUTPUTS).cast({"pts": pl.Int64})
    for name, (run, _) in engines.items():
        assert run(frame.head(20000)).select(OUTPUTS).cast({"pts": pl.Int64}).equals(expected), name

    latencies = {name: latency(score, calls=2000 if "interpreted" in name else 20000)
                 for name, (_, score) in engines.items()}
    results = {name: (rows_per_s(run, reps=1 if "interpreted" in name else 7), *latencies[name])
               for name, (run, _) in engines.items()}
    print(f"{'engine':<26}{'batch rows/s':>16}{'ns/row':>10}{'score p50 us':>14}{'score p99 us':>14}")
    for name, (batch, p50, p99) in results.items():
        print(f"{name:<26}{batch:>16,.0f}{1e9 / batch:>10.1f}{p50:>14.1f}{p99:>14.1f}")


print("12 score bands")
compare(BETWEEN)
print("bands AND a string list")
compare(GATED)
