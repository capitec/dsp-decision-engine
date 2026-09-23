"""score(): one record in, one dict out, in every mode; concurrent, retunable, fast."""
import gc
import threading
import time

import pytest

from decider import flow, missing_as, param
from decider.engine import Engine
from decider.engine.params import ParamsError


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def cap_by_income_band(term_cap: float, min_net_salary: float, cap: float = param(48.0, ge=6, le=60),
                       income_threshold: float = param(5000.0, ge=0)) -> float:
    return min(term_cap, cap) if min_net_salary < income_threshold else term_cap


flagship = flow(disposable_income, affordability_ratio, cap_by_income_band)
ROW = {"net_income": 4100.0, "expenses": 1500.0, "instalment": 800.0, "term_cap": 60.0, "min_net_salary": 4100.0}


def _expected(row, cap=48.0):
    ratio = (row["net_income"] - row["expenses"]) / row["instalment"]
    capped = min(row["term_cap"], cap) if row["min_net_salary"] < 5000.0 else row["term_cap"]
    return {"affordability_ratio": ratio, "cap_by_income_band": capped}


def test_score_returns_the_record_plus_every_output(bind):
    assert bind(flagship).score(ROW) == {**ROW, **_expected(ROW)}


def test_an_emit_is_seen_only_by_the_pipeline_that_asked_for_it(bind):
    assert bind(flagship.emit("disposable_income")).score(ROW)["disposable_income"] == 2600.0
    assert "disposable_income" not in bind(flagship).score(ROW)


def test_validity_and_fills_never_leak_between_calls(bind):
    def net(net_income: float | None, expenses: float = missing_as(100.0)) -> float:
        return (-1.0 if net_income is None else net_income) - expenses

    exe = bind(flow(net))
    full = {"net_income": 10.0, "expenses": 3.0}
    calls = [full, {"net_income": None, "expenses": 3.0}, full, {"net_income": 10.0}, full,
             {"net_income": 10.0, "expenses": None}, full, {}, full]
    assert [exe.score(r)["net"] for r in calls] == [7.0, -4.0, 7.0, -90.0, 7.0, -90.0, 7.0, -101.0, 7.0]


def test_a_bad_retune_is_a_hard_error(bind):
    exe = bind(flagship)
    with pytest.raises(ParamsError, match="cap_by_income_band: param 'cap'"):
        exe.score(ROW, params={"cap_by_income_band": {"cap": 999.0}})
    with pytest.raises(ParamsError, match="Did you mean 'cap_by_income_band'"):
        exe.score(ROW, params={"cap_by_income_bnad": {"cap": 24.0}})
    assert exe.score(ROW)["cap_by_income_band"] == 48.0


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_sixteen_threads_score_concurrently_with_no_cross_talk(mode):
    exe = Engine().bind(flagship, mode=mode)
    exe.score(ROW)
    n_threads, rounds = 16, 200
    start = threading.Barrier(n_threads)
    mismatches = [None] * n_threads

    def worker(t):
        rows = [{"net_income": 3000.0 + 100.0 * t + 7.0 * k, "expenses": 500.0 + 3.0 * t + k,
                 "instalment": 200.0 + t + k, "term_cap": 60.0 + t,
                 "min_net_salary": (3000.0 + 100.0 * t) if k % 2 else 9000.0 + t} for k in range(8)]
        expected = [_expected(r) for r in rows]
        start.wait()
        bad = 0
        for _ in range(rounds):
            for r, e in zip(rows, expected):
                out = exe.score(r)
                bad += (out["affordability_ratio"], out["cap_by_income_band"]) != tuple(e.values())
        mismatches[t] = bad

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert mismatches == [0] * n_threads


def test_fused_flagship_score_p50_is_within_60_microseconds():
    # Three rounds, so a load spike on a shared box can't fail a real ~30 µs result.
    exe = Engine().bind(flagship, mode="fused")
    for _ in range(500):
        exe.score(ROW)
    p50s = []
    for _ in range(3):
        gc.collect()
        samples = []
        for _ in range(2000):
            t0 = time.perf_counter()
            exe.score(ROW)
            samples.append(time.perf_counter() - t0)
        p50s.append(sorted(samples)[1000] * 1e6)
        if p50s[-1] <= 60.0:
            return
    pytest.fail(f"fused flagship score() p50 over 60 µs in every round: {[round(x, 1) for x in p50s]}")
