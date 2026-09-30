"""No-op overhead of in-kernel trace capture, measured through the real engine.

Baselines representative record / frame / branch / loop / table / tree paths in
interpreted, stepped and fused modes, then measures the cost of one extra int64
trace output per scalar step (the capture write) and of a tree's real in-kernel
`trace_output`. Exercise an override through `Session` for the record path.

    uv run python notes/vscode-redesign/experimentation/01-trace-capture/overhead.py
"""
import gc
import time

import numpy as np
import polars as pl

from decider import Engine, branch, flow, frame_step, loop, param, step
from decider.steps.tables import DecisionTableConfig
from decider.steps.trees import TreeConfig

N = 300_000
rng = np.random.default_rng(0)


def _time(f, *args):
    t = time.perf_counter()
    f(*args)
    return time.perf_counter() - t


def rows_per_s(run, n=N, reps=7):
    run(frame)
    return n / min(_time(run, frame) for _ in range(reps))


def latency(score, calls=20000):
    for _ in range(500):
        score(ROW)
    gc.collect()
    samples = sorted(_time(score, ROW) for _ in range(calls))
    return samples[len(samples) // 2] * 1e6, samples[int(len(samples) * 0.99)] * 1e6


def add(x):
    return x + 1.0


@step(outputs=("v", "tr"))
def add_traced(x) -> tuple[float, int]:
    return x + 1.0, 7


def build_chain_one_trace():
    steps = [step(add, output="v0").named("a0")]
    for k in range(1, 10):
        steps.append(step(add, output=f"v{k}").relabel(reads={"x": f"v{k-1}"}).named(f"a{k}"))
    steps.append(step(lambda v9: int(v9), output="trace").named("tr"))
    return flow(*steps, name="chain")


def build_chain(with_trace):
    if with_trace:
        steps = [add_traced.relabel(writes={"v": "v0", "tr": "tr0"}).named("a0")]
        for k in range(1, 10):
            steps.append(add_traced.relabel(
                reads={"x": f"v{k-1}"}, writes={"v": f"v{k}", "tr": f"tr{k}"}).named(f"a{k}"))
    else:
        steps = [step(add, output="v0").named("a0")]
        for k in range(1, 10):
            steps.append(step(add, output=f"v{k}").relabel(reads={"x": f"v{k-1}"}).named(f"a{k}"))
    return flow(*steps, name="chain")


def build_branch_loop():
    def eligible(age: int) -> bool:
        return age >= 18

    @step(output="limit")
    def full_limit(income: float, multiple: float = param(3.0)) -> float:
        return income * multiple

    @step(output="limit")
    def no_limit(income: float) -> float:
        return 0.0

    def unpaid(balance: float) -> bool:
        return balance > 0.0

    @step(output="balance")
    def pay(balance: float, instalment: float, rate: float = param(0.01)) -> float:
        return balance * (1 + rate) - instalment

    @step(output="months")
    def tick(months: int) -> int:
        return months + 1

    return flow(
        branch(eligible, full_limit, no_limit, modifies=["limit"], name="by_eligibility"),
        loop(unpaid, flow(pay, tick, name="month"), carries=["balance", "months"],
             max_iterations=360, name="repay"),
        name="offer")


def build_tree(trace: str | None):
    nodes, edges = [], []
    for d in range(5):
        lo = (1 << (d + 1)) - 1
        for k in range(1 << d):
            nid = f"n{lo + k}"
            if d == 4:
                nodes.append({"id": nid, "data": {"type": "leaf", "result_idx": k % 4}})
            else:
                nodes.append({"id": nid, "data": {"type": "unary", "condition": {
                    "op": ">=", "feature": "f0", "threshold": 0.5}}})
                edges += [{"source": nid, "target": f"n{2 * (lo + k) + 1}", "data": {"sourceIndex": [0]}},
                          {"source": nid, "target": f"n{2 * (lo + k) + 2}", "data": {"sourceIndex": [1]}}]
    doc = {"name": "risk", "nodes": nodes, "edges": edges,
           "output": {"data": [{"band": i} for i in range(4)],
                      "default": {"band": -1}, "dtypes": [["band", "Int64"]]}}
    kwargs = {"tree": doc}
    if trace:
        kwargs[trace] = "risk_" + trace
    return TreeConfig(name="risk", **kwargs)


def build_table():
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "band",
        "columns": {"lo": "Float64", "hi": "Float64", "band": "String"},
        "rows": [{"lo": None, "hi": 600.0, "band": "C"}, {"lo": 600.0, "hi": 700.0, "band": "B"},
                 {"lo": 700.0, "hi": None, "band": "A"}],
        "expression": {"type": "between", "variable": "bureau_score",
                       "lower_bound_column": "lo", "upper_bound_column": "hi"},
        "outputs": ["band"], "default": ["C"]})


def build_frame():
    @frame_step(reads=["client_id"], writes=["prior_defaults"])
    def join_history(df: pl.DataFrame) -> pl.DataFrame:
        return df.join(pl.DataFrame({"client_id": [1, 2], "prior_defaults": [2, 0]}),
                       on="client_id", how="left")

    return flow(join_history, name="history")


def frames():
    return {
        "chain": pl.DataFrame({"x": rng.uniform(0, 1000, N)}),
        "offer": pl.DataFrame({"client_id": [1, 2] * (N // 2), "age": rng.integers(15, 60, N),
                               "income": rng.uniform(500, 5000, N), "balance": rng.uniform(0, 2000, N),
                               "instalment": rng.uniform(50, 300, N), "months": np.zeros(N, int)}),
        "tree": pl.DataFrame({"f0": rng.uniform(0, 1, N)}),
        "table": pl.DataFrame({"bureau_score": rng.uniform(300, 900, N)}),
        "frame": pl.DataFrame({"client_id": [1, 2] * (N // 2)}),
    }


def report(name, exe, key):
    global frame, ROW
    frame = frames()[key]
    ROW = frame.row(0, named=True)
    batch = rows_per_s(exe.run)
    p50, p99 = latency(exe.score, calls=500 if key == "tree" else 20000)
    print(f"{name:<28}{batch:>14,.0f}{p50:>12.1f}{p99:>12.1f}")


if __name__ == "__main__":
    global frame, ROW

    print("=== scalar chain: no trace vs one int64 trace output per step ===")
    print(f"{'engine':<28}{'batch rows/s':>14}{'score p50 us':>12}{'score p99 us':>12}")
    for label, build, mode in (("fused no-trace", lambda: build_chain(False), "fused"),
                               ("fused +1 int64 col", lambda: build_chain_one_trace(), "fused"),
                               ("fused +trace col", lambda: build_chain(True), "fused"),
                               ("stepped +trace col", lambda: build_chain(True), "stepped"),
                               ("interpreted +trace col", lambda: build_chain(True), "interpreted")):
        exe = Engine().bind(build(), mode=mode)
        report(label, exe, "chain")

    print("\n=== branch + loop ===")
    print(f"{'engine':<28}{'batch rows/s':>14}{'score p50 us':>12}{'score p99 us':>12}")
    for mode in ("fused", "stepped", "interpreted"):
        exe = Engine().bind(build_branch_loop(), mode=mode)
        report(f"{mode} branch+loop", exe, "offer")

    print("\n=== tree: no trace vs path_output vs trace_output (in-kernel) ===")
    print(f"{'engine':<28}{'batch rows/s':>14}{'score p50 us':>12}{'score p99 us':>12}")
    for label, trace in (("no trace", None), ("path_output", "path_output"),
                         ("trace_output", "trace_output")):
        exe = Engine().bind(build_tree(trace), mode="fused")
        report(f"fused {label}", exe, "tree")

    print("\n=== table and frame ===")
    print(f"{'engine':<28}{'batch rows/s':>14}{'score p50 us':>12}{'score p99 us':>12}")
    for mode in ("fused", "interpreted"):
        report(f"{mode} table", Engine().bind(build_table(), mode=mode), "table")
    for mode in ("fused", "interpreted"):
        report(f"{mode} frame", Engine().bind(build_frame(), mode=mode), "frame")

    print("\n=== override (record path, Session) ===")
    exe = Engine().bind(build_chain(False), mode="stepped")
    frame = frames()["chain"]
    s = exe.session(frame)
    s.break_at("chain/a1")
    s.resume()
    s.set("v0", 1.0)
    s.resume()
    assert s.output()["v9"].to_list()[0] == 10.0
    print(f"override path: set('v0') recorded as override@chain/v0; {len(s.events)} session events emitted")
