"""Where one request's microseconds go inside the process, and the kernel floor.

Every term of `RequestHandler.process_fn` is timed on its own; `score()` is
timed as cumulative prefixes of the work it does, so each stage is the next
difference. The whole table runs `--reps` times and the lowest p50 and p99 of
each term is reported: this box is shared, and the minimum is what the term
costs when nothing preempts it. HTTP framing is measured over a socket in
`benchmarks/serving_transports.py`.

    uv run python benchmarks/serving_latency.py [flagship|tree|gated|all] [--reps 3]
"""
import asyncio
import gc
import io
import json
import sys
import tempfile
import time

import numpy as np
import polars as pl
from pydantic_core import from_json, to_json

from decider import flow, param
from decider.config import JsonFileStore
from decider.engine.run.engine import _NO_FRAME, _load
from decider.engine.run.state import State
from decider.serving.handler import RequestHandler
from decider.serving.parse import coerce_record, parse_application_json

sys.path.insert(0, "benchmarks")
import tree_walk  # noqa: E402  the tree document the tree benchmark uses

CALLS = 20_000
WARM = 2_000


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def cap_by_income_band(term_cap: float, min_net_salary: float, cap: float = param(48.0, ge=6, le=60),
                       income_threshold: float = param(5000.0, ge=0)) -> float:
    if min_net_salary < income_threshold:
        return min(term_cap, cap)
    return term_cap


FLAGSHIP_ROW = {"net_income": 4100.0, "expenses": 1500.0, "instalment": 800.0,
                "term_cap": 60.0, "min_net_salary": 4100.0}
TREE_ROW = {k: v for k, v in tree_walk.ROW.items() if k != "channel"}


def cases():
    from decider.steps.trees import TreeConfig

    return {
        "flagship": (flow(disposable_income, affordability_ratio, cap_by_income_band), FLAGSHIP_ROW),
        "tree": (TreeConfig(name="campaign", tree=tree_walk.DOC, feature_types=tree_walk.TYPES), TREE_ROW),
        "gated": (TreeConfig(name="campaign", tree=tree_walk._gated(tree_walk.DOC),
                             feature_types=tree_walk.TYPES), tree_walk.ROW),
    }


def _ipc(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_ipc_stream(buf)
    return buf.getvalue()


def _pct(samples):
    samples.sort()
    n = len(samples)
    return samples[n // 2] * 1e6, samples[int(n * 0.99)] * 1e6


def stats(fn, calls=CALLS, warm=WARM):
    """p50 and p99 of `fn()` in µs."""
    for _ in range(warm):
        fn()
    gc.collect()
    samples = np.empty(calls)
    perf = time.perf_counter
    for k in range(calls):
        t = perf()
        fn()
        samples[k] = perf() - t
    return _pct(samples)


async def _atime(fn, calls, warm):
    for _ in range(warm):
        await fn()
    gc.collect()
    samples = np.empty(calls)
    perf = time.perf_counter
    for k in range(calls):
        t = perf()
        await fn()
        samples[k] = perf() - t
    return _pct(samples)


async def _noop():
    return None


def _gc_off(fn):
    def run():
        gc.disable()
        try:
            return fn()
        finally:
            gc.enable()
    return run


def terms(pipeline, row, loop):
    """(name, is_async, callable) for every term of one request, plus the floors."""
    store = JsonFileStore(basepath=tempfile.mkdtemp(dir=".scratch"))
    store.create_version({"params": {}})
    handler = RequestHandler(store, pipeline, mode="fused")
    handler.stage()
    handler.activate()
    live = handler.module_fn()
    exe, params, dates = live.executable, live.params, live.dates
    plan, runner = exe.plan, exe.runner
    body = json.dumps(row).encode()
    result = exe.score(row, params)

    def build():
        state = State(plan, _NO_FRAME, 1)
        for dtype, versions in exe._inputs:
            _load(state, row, versions, dtype)
        return state

    def prepared():
        return build(), exe._params(params, 1)

    def ran():
        state, run = prepared()
        for _ in runner.iterate(plan, state, run):
            pass
        return state

    def scored():
        state = ran()
        out = {k: x for k, x in row.items() if k not in exe._hidden}
        for k, v in exe._results:
            values, valid = state.read(v)
            out[k] = None if valid is not None and not valid[0] else values.tolist()[0]
        return out

    ipc = _ipc(pl.DataFrame([row]))
    schema_msg, batch_msg = ipc[:8 + int.from_bytes(ipc[4:8], "little")], ipc[8 + int.from_bytes(ipc[4:8], "little"):]
    one = pl.read_ipc_stream(ipc)
    rows = [
        ("json.loads(body)", 0, lambda: json.loads(body)),
        ("from_json(body)  [not used today]", 0, lambda: from_json(body)),
        ("input_fn body (sync part)", 0, lambda: parse_application_json(body)),
        ("coerce_record", 0, lambda: coerce_record(row, dates)),
        ("module_fn", 0, handler.module_fn),
        ("score() total", 0, lambda: exe.score(row, params)),
        ("to_json(result)", 0, lambda: to_json(result)),
        ("output_fn", 0, lambda: handler.output_fn(result, "application/json")),
        ("await an empty coroutine", 1, _noop),
        ("await input_fn", 1, lambda: handler.input_fn(body, "application/json")),
        ("await process_fn (no HTTP)", 1, lambda: handler.process_fn(body, "application/json", "application/json")),
        ("same work, sync, one function", 0, lambda: handler.output_fn(
            exe.score(coerce_record(parse_application_json(body), dates), params), "application/json")),
        ("same work, sync, gc off", 0, _gc_off(lambda: handler.output_fn(
            exe.score(coerce_record(parse_application_json(body), dates), params), "application/json"))),
        ("|score: State(plan, _NO_FRAME, 1)", 0, lambda: State(plan, _NO_FRAME, 1)),
        ("|score: + _load inputs", 0, build),
        ("|score: + _params bundle", 0, prepared),
        ("|score: + runner.iterate", 0, ran),
        ("|score: + output dict", 0, scored),
    ]
    rows += [
        (f"|arrow: read_ipc_stream ({len(ipc)} B)", 0, lambda: pl.read_ipc_stream(ipc)),
        (f"|arrow: read_ipc_stream, schema cached ({len(batch_msg)} B)", 0,
         lambda: pl.read_ipc_stream(schema_msg + batch_msg)),
        ("|arrow: + df.row(0, named=True)", 0, lambda: pl.read_ipc_stream(ipc).row(0, named=True)),
        ("|arrow: write_ipc_stream of the result", 0, lambda: _ipc(pl.DataFrame([result]))),
        ("|arrow: exe.run(one-row frame)", 0, lambda: exe.run(one, params)),
    ]
    state = ran()
    run = exe._params(params, 1)
    rows += _floors(exe, state, run, row, result)
    calls = 4_000 if len(plan.calls) > 20 else CALLS
    return rows, calls, len(body)


def _floors(exe, state, run, row, result):
    """The reuse-everything floor: kernels alone, then a whole request with no per-call allocation."""
    units = _units(exe, state, run)
    if not units:
        return []
    # A kernel reading a str/bytes column is handed a representation the runner builds per call,
    # not the object array State holds, so there is no buffer to pre-bind: no floor for that case.
    if any(a.dtype == object for _, values, _, _ in units for a in values.values()):
        return []
    # Writable copies: `_load` freezes its blocks, and the floor writes inputs in place.
    own = {vid: np.array(a) for _, values, _, _ in units for vid, a in values.items()}
    args = []
    for unit, values, valid, bundles in units:
        cols = tuple([own[v.id] for v in unit.reads])
        ones = np.ones(1, np.bool_) if unit.optional else None
        valids = tuple([valid.get(v.id, ones) for v in unit.optional])
        params: list = []
        for cid, has_params, consts, is_row in unit._layout:
            bundle = bundles[cid] if has_params else ()
            params += (bundle, consts) if is_row else [*bundle, *consts]
        outs = tuple([np.empty(1, dtype) for _, dtype in unit.writes]
                     + [np.empty(1, np.bool_) for _ in unit._masked])
        args.append((unit.fn, cols, valids, tuple(params), outs))

    def dispatch():
        for fn, cols, valids, params, outs in args:
            fn(1, cols, valids, params, outs)

    def via_run():
        for unit, values, valid, bundles in units:
            unit.run(dict(values), valid, bundles, 1)

    rows = [("floor: kernel dispatchers only", 0, dispatch),
            ("floor: unit.run (+ per-call allocs)", 0, via_run)]

    # Inputs written into the arrays the kernels already hold, outputs read out of theirs.
    slots = [(v.name, own[v.id]) for _, versions in exe._inputs for v in versions if v.name in row and v.id in own]
    result_slots = [(k, a) for k, v in exe._results for a in [own.get(v.id)] if a is not None]
    if len(slots) != len(row):
        return rows
    if len(result_slots) != len(exe._results):
        return rows
    out = dict(result)
    body = json.dumps(row).encode()

    def whole():
        record = from_json(body)
        for name, arr in slots:
            arr[0] = record[name]
        for fn, cols, valids, params, outs in args:
            fn(1, cols, valids, params, outs)
        for k, arr in result_slots:
            out[k] = arr[0].item()
        return to_json(out)

    return rows + [("floor: whole request, nothing allocated", 0, whole)]


def _units(exe, state, run):
    runner = exe.runner
    seen, out = set(), []
    for unit in list(getattr(runner, "units", {}).values()) + list(getattr(runner, "packed", {}).values()):
        if id(unit) in seen or not hasattr(unit, "_layout"):
            continue
        seen.add(id(unit))
        if not {v.id for v in unit.reads} <= state.values.keys():
            continue
        values = {v.id: state.values[v.id] for v in unit.reads}
        for v, dtype in unit.writes:
            values.setdefault(v.id, np.empty(1, dtype))
        out.append((unit, values, {}, {c.id: runner._bundle(c.id, run, 1) for c in unit.calls if c.node.params}))
    return out


def main(which, reps):
    loop = asyncio.new_event_loop()
    print(f"timer overhead p50: {stats(lambda: None)[0]:.3f} µs, reps: {reps}\n")
    for name in which:
        pipeline, row = cases()[name]
        rows, calls, nbytes = terms(pipeline, row, loop)
        best = {}
        for _ in range(reps):
            for term, is_async, fn in rows:
                p = loop.run_until_complete(_atime(fn, calls, WARM)) if is_async else stats(fn, calls, WARM)
                old = best.get(term)
                best[term] = p if old is None else (min(old[0], p[0]), min(old[1], p[1]))
        print(f"── {name}: {len(row)} inputs, {nbytes} byte JSON body, {calls} calls/rep ──")
        print(f"{'term':<38}{'p50 µs':>9}{'p99 µs':>9}{'stage':>9}")
        last = 0.0
        for term, _, _ in rows:
            p50, p99 = best[term]
            stage = ""
            if term.startswith("|score:"):
                stage = f"{p50 - last:>9.2f}"
                last = p50
            print(f"{term:<38}{p50:>9.2f}{p99:>9.2f}{stage}")
        print()
    loop.close()


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    reps = int(argv[1]) if "--reps" in sys.argv and len(argv) > 1 else 3
    arg = argv[0] if argv else "all"
    main(list(cases()) if arg == "all" else [arg], reps)
