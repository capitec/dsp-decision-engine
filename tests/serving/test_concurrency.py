"""One shared `Executable` answers correctly when many threads call it at once, in every mode.

Races are probabilistic: a failure here is a real bug, a pass is only evidence. The interpreter
switches threads every microsecond so the interleavings are forced, and every answer is compared
with the one a single thread gets.
"""
import asyncio
import importlib
import sys
import threading
import typing as t
import warnings
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import polars as pl
import pytest
from numba.core.errors import NumbaError

from decider import ConfigurableStep, Engine, branch, each, flow, missing_as, param, step
from decider.ops import Text
from decider.config import JsonFileStore
from decider._engine.compile.njit import jit
from decider._engine.compile.rows import rows_class
from decider._engine.params import ParamsCache
from decider._engine.trace import TraceSink
from decider.serving._handler import Live, RequestHandler

MODES = ("interpreted", "stepped", "fused")
THREADS = 16


@pytest.fixture(autouse=True)
def fast_thread_switching():
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(old)


def together(fn, items):
    """`fn(item)` for every item on its own thread, all released at once; results in order."""
    barrier = threading.Barrier(len(items))

    def go(item):
        barrier.wait()
        return fn(item)

    with ThreadPoolExecutor(len(items)) as pool:
        return list(pool.map(go, items))


def hammer(fn, items):
    """`fn(item)` for every item on a pool of threads; results in order."""
    with ThreadPoolExecutor(THREADS) as pool:
        return list(pool.map(fn, items))


def outcome(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        return type(e).__name__, str(e)


# --- the pipeline most cases share ---------------------------------------------------


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def is_private(sector_code: int) -> bool:
    return sector_code == 1


@step(output="term_cap")
def cap_private(term_cap: float, cap: float = param(54.0)) -> float:
    return min(term_cap, cap)


@step(output="term_cap")
def cap_public(term_cap: float, cap: float = param(60.0)) -> float:
    return min(term_cap, cap)


def pipeline():
    sector = branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by_sector")
    return flow(disposable_income, affordability_ratio, sector, name="p")


def record(i):
    return {"net_income": 4000.0 + i, "expenses": 1500.0, "instalment": 800.0 + i % 7,
            "term_cap": 72.0 + i % 5, "sector_code": 1 + i % 2}


RECORDS = [record(i) for i in range(THREADS * 2)]


@pytest.fixture(scope="module")
def serial():
    exe = Engine().bind(pipeline(), mode="interpreted")
    return [exe.score(r) for r in RECORDS]


# --- (a) concurrent first calls on a cold executable ---------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_concurrent_first_calls_on_a_cold_executable_agree_with_one_thread(mode, serial):
    for _ in range(25):
        exe = Engine().bind(pipeline(), mode=mode)
        got = together(exe.score, RECORDS[:THREADS])
        assert got == serial[:THREADS]
        assert hammer(exe.score, RECORDS) == serial


@pytest.mark.parametrize("mode", MODES)
def test_concurrent_first_frame_runs_on_a_cold_executable_agree_with_one_thread(mode):
    frame = pl.DataFrame(RECORDS)
    expected = Engine().bind(pipeline(), mode="interpreted").run(frame)
    for _ in range(15):
        exe = Engine().bind(pipeline(), mode=mode)
        assert all(out.equals(expected) for out in together(lambda _: exe.run(frame), range(THREADS)))


# --- (b) str params on the span path -------------------------------------------------


def is_sector(sector: Text, want: str = param("privé")) -> bool:
    return sector == want


def span_pipeline():
    return flow(is_sector, name="s")


def span_words(rep):
    # Long, and new every round, so each round meets a cold cache entry whose freed copy gets reused.
    return {"a": f"privé{rep}" * 600, "b": f"café{rep}" * 700}


@pytest.mark.parametrize("mode", ("stepped", "fused"))
def test_str_params_on_the_span_path_survive_concurrent_first_use(mode):
    exe = Engine().bind(span_pipeline(), mode=mode)
    exe.score({"sector": "x"})
    for rep in range(40):
        words = span_words(rep)
        calls = [({"s": {"is_sector": {"want": words["ab"[i % 2]]}}},
                  words["ab"[i % 2]] if i % 3 else words["ba"[i % 2]]) for i in range(THREADS * 2)]
        expected = [sector == doc["s"]["is_sector"]["want"] for doc, sector in calls]

        def score(c):
            return exe.score({"sector": c[1]}, c[0])["is_sector"]

        assert together(score, calls[:THREADS]) == expected[:THREADS]
        assert hammer(score, calls) == expected


# --- (c) traced and untraced calls overlap -------------------------------------------


def trace_of(sink):
    return sorted((e.kind.name, e.step, e.arm, e.iteration, e.record, e.value) for e in sink.events())


@pytest.mark.parametrize("mode", MODES)
def test_each_traced_call_gets_exactly_its_own_events(mode, serial):
    exe = Engine().bind(pipeline(), mode=mode)
    expected = []
    for r in RECORDS:
        sink = TraceSink()
        exe.score(r, trace=sink)
        expected.append(trace_of(sink))
    assert all(expected)

    def call(i):
        if i % 2:
            return exe.score(RECORDS[i]), None
        sink = TraceSink()
        return exe.score(RECORDS[i], trace=sink), trace_of(sink)

    for _ in range(30):
        for i, (out, events) in enumerate(hammer(call, range(len(RECORDS)))):
            assert out == serial[i]
            assert events is None or events == expected[i]


# --- (d) the latest report is per thread ---------------------------------------------


@pytest.mark.parametrize("mode", ("interpreted", "stepped"))
def test_report_is_the_latest_call_on_the_calling_thread(mode):
    exe = Engine(params_validation="lazy").bind(pipeline(), mode=mode)
    counter = iter(range(10**9))

    def call(sector_code):
        # A document nobody has used before, so validation runs and the report lists the node it reached.
        n = next(counter)
        doc = {"p": {"by_sector": {"cap_private": {"cap": 50.0 + n}, "cap_public": {"cap": 80.0 + n}}}}
        exe.score({**RECORDS[0], "sector_code": sector_code}, doc)
        return exe.report.validated

    private, public = call(1), call(2)
    assert private != public and private and public
    for _ in range(20):
        got = hammer(call, [1 + i % 2 for i in range(THREADS * 2)])
        assert got == [private if i % 2 == 0 else public for i in range(THREADS * 2)]


def test_a_thread_that_has_not_called_yet_has_an_empty_report():
    exe = Engine().bind(pipeline(), mode="interpreted")
    exe.score(RECORDS[0])
    with ThreadPoolExecutor(1) as pool:
        assert pool.submit(lambda: exe.report.validated).result() == []


def test_a_param_in_the_record_warns_once_however_many_threads_see_it():
    for _ in range(20):
        exe = Engine().bind(pipeline(), mode="interpreted")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            together(lambda i: exe.score({**RECORDS[i], "cap": 1.0}), range(THREADS))
        assert len([w for w in caught if "'cap' is a param" in str(w.message)]) == 1


# --- (e) a version is staged and activated while requests run ------------------------


def _scale(x: float, factor: float) -> float:
    return x * factor


def capped(bonus: float, cap: float = param(5000.0)) -> float:
    return min(bonus, cap)


class Bonus(ConfigurableStep):
    type: t.Literal["concurrency_bonus"] = "concurrency_bonus"
    column: str
    factor: float = 1.0

    def to_ir(self, ctx):
        return ctx.call(self, _scale, inputs={"x": self.column}, values={"factor": self.factor})


def build(bonus: ConfigurableStep):
    return flow(disposable_income, bonus, capped, name="afford").emit("bonus")


V0 = {"bonus": {"type": "concurrency_bonus", "name": "bonus", "column": "disposable_income", "factor": 1.0},
      "params": {"afford": {"capped": {"cap": 5000.0}}}}
V1 = {"bonus": {**V0["bonus"], "factor": 2.0}, "params": {"afford": {"capped": {"cap": 3000.0}}}}
ANSWERS = {(6000.0, 5000.0), (12000.0, 3000.0)}


def test_stage_and_activate_under_load_never_serve_a_wrong_answer(tmp_path):
    store = JsonFileStore(basepath=str(tmp_path))
    store.create_version(V0)
    store.create_version(V1)
    handler = RequestHandler(store, build)
    handler.stage("0.0.0")
    handler.activate()
    stop = threading.Event()
    wrong = []

    def serve():
        while not stop.is_set():
            live = handler.module_fn()
            out = live.executable.score({"net_income": 9000.0, "expenses": 3000.0}, live.params)
            if (out["bonus"], out["capped"]) not in ANSWERS:
                wrong.append(out)

    with ThreadPoolExecutor(THREADS) as pool:
        futures = [pool.submit(serve) for _ in range(THREADS - 1)]
        for version in ("0.1.0", "0.0.0", "0.1.0"):
            handler.stage(version)
            handler.activate()
        handler.rollback()
        stop.set()
        for f in futures:
            f.result()
    assert not wrong


def test_process_fn_under_load_while_a_version_is_activated(tmp_path):
    store = JsonFileStore(basepath=str(tmp_path))
    store.create_version(V0)
    store.create_version(V1)
    handler = RequestHandler(store, build)
    handler.stage("0.0.0")
    handler.activate()
    body = b'{"net_income": 9000.0, "expenses": 3000.0}'

    def request(_):
        return asyncio.run(handler.process_fn(body, "application/json", "application/json")).content

    with ThreadPoolExecutor(THREADS) as pool:
        futures = [pool.submit(request, i) for i in range(THREADS * 8)]
        handler.stage("0.1.0")
        handler.activate()
        bodies = {f.result() for f in futures}
    assert len(bodies) <= 2 and all(b'"bonus"' in b for b in bodies)


# --- (f) a packed branch hits ArmOutOfRange mid-load ----------------------------------


def band_of(band: int) -> int:
    return band


@step(output="price")
def band0(price: float) -> float:
    return 1.0 + 0.0 * price


@step(output="price")
def band1(price: float) -> float:
    return 2.0 + 0.0 * price


def banded():
    return flow(branch(band_of, band0, band1, modifies=["price"], name="rules"), name="b")


def test_arm_out_of_range_under_load_raises_for_those_rows_only_and_keeps_the_kernel():
    records = [{"band": i % 3, "price": float(i)} for i in range(THREADS * 4)]
    serial = Engine().bind(banded(), mode="interpreted")
    expected = [outcome(serial.score, r) for r in records]
    assert any(isinstance(e, tuple) for e in expected) and any(isinstance(e, dict) for e in expected)
    for _ in range(15):
        exe = Engine().bind(banded(), mode="fused")
        assert hammer(lambda r: outcome(exe.score, r), records) == expected
        assert len(exe.runner.packed) == 1


def test_a_packed_kernel_numba_refuses_is_dropped_once_and_every_call_still_answers(serial):
    def refuse(*args, **kwargs):
        raise NumbaError("refused")

    for _ in range(15):
        exe = Engine().bind(pipeline(), mode="fused")
        exe.score(RECORDS[0])
        (path,) = exe.runner.packed
        with patch.object(type(exe.runner.packed[path]), "run", refuse):
            assert hammer(exe.score, RECORDS) == serial
        assert exe.runner.packed == {}
        assert exe.score(RECORDS[1]) == serial[1]


# --- (g) each ------------------------------------------------------------------------


def heavy(weight: float = missing_as(0.0), heavy_kg: float = param(20.0)) -> bool:
    return weight > heavy_kg


def orders():
    return flow(each("items", flow(heavy, name="item"), name="items"), name="order")


@pytest.mark.parametrize("mode", MODES)
def test_each_under_concurrent_first_calls_agrees_with_one_thread(mode):
    carts = [{"items": [{"weight": float(w)} for w in range(i % 5 + 1)] + [{"weight": 25.0}]}
             for i in range(THREADS * 2)]
    expected = [Engine().bind(orders(), mode="interpreted").score(c) for c in carts]
    for _ in range(15):
        exe = Engine().bind(orders(), mode=mode)
        assert together(exe.score, carts[:THREADS]) == expected[:THREADS]
        assert hammer(exe.score, carts) == expected


# --- (h) a session opened on the shared executable mid-load ---------------------------


def test_a_session_opened_on_a_shared_executable_neither_disturbs_nor_is_disturbed(serial):
    exe = Engine().bind(pipeline(), mode="fused")
    frame = pl.DataFrame(RECORDS)
    expected = Engine().bind(pipeline(), mode="interpreted").run(frame)
    stop = threading.Event()
    wrong = []

    def serve(k):
        i = 0
        while not stop.is_set():
            i = (i + 1) % len(RECORDS)
            if exe.score(RECORDS[i]) != serial[i]:
                wrong.append((k, i))

    with ThreadPoolExecutor(THREADS) as pool:
        futures = [pool.submit(serve, k) for k in range(THREADS - 1)]
        for _ in range(10):
            s = exe.session(frame)
            s.resume()
            assert s.output().equals(expected)
        stop.set()
        for f in futures:
            f.result()
    assert not wrong


# --- shared build caches: one thing built per key, however many threads ask ----------


_fresh = iter(range(1000, 10**9))


def scaled_by(k):
    def scale(x: float) -> float:
        return x * k

    return scale


def test_concurrent_jit_of_one_function_gives_one_dispatcher():
    for _ in range(15):
        k = float(next(_fresh))
        dispatchers = together(lambda f: jit(f)[1], [scaled_by(k) for _ in range(THREADS)])
        assert len({id(d) for d in dispatchers}) == 1


def test_concurrent_row_layouts_of_one_schema_are_one_class():
    for _ in range(15):
        schema = ((f"price_{next(_fresh)}", float),)
        assert len(set(together(lambda _: rows_class(schema), range(THREADS)))) == 1


@pytest.mark.parametrize("mode", ("stepped", "fused"))
def test_cold_executables_sharing_a_kernel_compile_at_once_and_agree(mode):
    for _ in range(8):
        k = float(next(_fresh))
        exes = [Engine().bind(flow(scaled_by(k), name=f"p{i}"), mode=mode) for i in range(THREADS)]
        assert together(lambda e: e.score({"x": 2.0})["scale"], exes) == [2.0 * k] * THREADS


def test_concurrent_first_validations_of_one_document_share_one_result():
    exe = Engine().bind(pipeline(), mode="interpreted")
    node = next(iter(exe.nodes.values()))
    for i in range(30):
        doc = {"p": {"by_sector": {"cap_private": {"cap": 50.0 + i}, "cap_public": {"cap": 80.0 + i}}}}
        cache = ParamsCache()
        key = cache.key(doc)
        assert len({id(v) for v in together(lambda _: cache.validate(key, doc, node), range(THREADS))}) == 1


def test_registry_resolves_a_lazy_module_and_lists_while_classes_register(tmp_path, monkeypatch):
    (tmp_path / "conc_root.py").write_text(
        "from decider._registry import BaseRegistryModule\n\n"
        "class Root(BaseRegistryModule, root=True):\n    pass\n")
    (tmp_path / "conc_leaf.py").write_text(
        "import typing as t\nfrom conc_root import Root\n\n"
        "class Leaf(Root):\n    type: t.Literal['leaf'] = 'leaf'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    root = importlib.import_module("conc_root").Root
    root.lazy({"leaf": "conc_leaf"})
    stop = threading.Event()

    def define():
        for i in range(100):
            type(f"Extra{i}", (root,), {"__annotations__": {"type": t.Literal[f"extra_{i}"]}, "type": f"extra_{i}"})
        stop.set()

    def ask(_):
        while not stop.is_set():
            with pytest.raises(LookupError):
                root.resolve("nope")

    try:
        with ThreadPoolExecutor(THREADS) as pool:
            futures = [pool.submit(ask, i) for i in range(THREADS - 1)] + [pool.submit(define)]
            for f in futures:
                f.result()
        classes = together(lambda _: root.resolve("leaf"), range(THREADS))
        assert len(set(classes)) == 1 and classes[0].__name__ == "Leaf"
    finally:
        for name in ("conc_root", "conc_leaf"):
            sys.modules.pop(name, None)


def test_the_default_dates_of_a_live_version_cannot_be_changed():
    live = Live(None, None, None)
    with pytest.raises(TypeError):
        live.dates["when"] = None
