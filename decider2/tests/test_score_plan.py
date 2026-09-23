"""BOUNDARY-REWORK.md Stage 4 — the single-record plan.

`Pipeline.score()` runs a `decider2.runtime.plan.ScorePlan` built once per
pipeline and held. These tests are the stage's acceptance bar, in the order
the brief ranks them: a wrong answer served from a stale or shared plan is
worse than any speed (doc 03 §2.1), so invalidation and thread safety come
first, then the retune contract (doc 08 §2), then the latency spec itself.

(No `from __future__ import annotations` here on purpose: the OPTIONAL
steps below need a real `float | None` annotation for harvesting to see
tier 3, not a string.)
"""
import gc
import threading
import time

import polars as pl
import pytest

from decider2 import flow, missing_as, module, param
from decider2.examples.flagship import (
    affordability_ratio,
    cap_by_income_band,
    disposable_income,
    pipeline as flagship,
)
from decider2.runtime import invoke
from decider2.runtime.plan import ScorePlan
from decider2.testing.equivalence import assert_equivalent
from decider2.testing.recompile import count_new_compiles
from decider2.types import Decision

ROW = {"net_income": 4100.0, "expenses": 1500.0, "instalment": 800.0, "term_cap": 60.0, "min_net_salary": 4100.0}


def _expected(row: dict, cap: float = 48.0, income_threshold: float = 5000.0) -> dict:
    """The flagship's answer, in plain Python — the oracle every threaded
    and pooled call below is checked against."""
    disp = row["net_income"] - row["expenses"]
    ratio = disp / row["instalment"]
    capped = min(row["term_cap"], cap) if row["min_net_salary"] < income_threshold else row["term_cap"]
    return {"affordability_ratio": ratio, "cap_by_income_band": capped}


def _fresh_flagship():
    return flow(disposable_income, affordability_ratio, cap_by_income_band)


# --- what is cached, and that it is ------------------------------------------


def test_plan_interface_and_flat_shape_are_built_once_and_held():
    p = _fresh_flagship()
    assert p.score_plan() is p.score_plan()
    assert p.interface is p.interface
    assert p.flatten_for_runtime() is p.flatten_for_runtime()
    p.score(ROW)
    assert p.score_plan() is p.score_plan(), "a call must not rebuild the plan"
    assert isinstance(p.score_plan(), ScorePlan)


def test_plan_holds_exactly_the_schema_invariant_pieces():
    plan = _fresh_flagship().score_plan()
    assert [s.name for s in plan.slots] == ["net_income", "expenses", "instalment", "term_cap", "min_net_salary"]
    assert plan.terminal_names == frozenset({"affordability_ratio", "cap_by_income_band"})
    assert [sp.module for sp in plan.params.spaces] == ["cap_by_income_band"]
    assert plan.mode.name == "fused"


def test_direct_invoke_score_and_pipeline_score_run_the_same_plan_code():
    p = _fresh_flagship()
    steps, group_ids, owners, spaces = p.flatten_for_runtime()
    direct = invoke.score(
        steps, ROW, interface=p.interface, group_ids=group_ids, owners=owners, param_spaces=spaces,
    )
    assert direct == p.score(ROW) == {**ROW, **_expected(ROW)}


# --- 1. invalidation ---------------------------------------------------------


def test_every_public_mutator_yields_a_pipeline_with_its_own_plan():
    """`.emit()`, `.drop()`, `.on_missing_input()`, `|` and `Module.bind()`
    all return a NEW pipeline (doc 03: "they do not mutate"); each must
    get its own plan, and the original's must be untouched."""
    p = _fresh_flagship()
    base_plan = p.score_plan()
    assert p.score(ROW) == {**ROW, **_expected(ROW)}

    emitted = p.emit("disposable_income")
    assert emitted.score_plan() is not base_plan
    assert "disposable_income" in emitted.score_plan().terminal_names
    assert emitted.score(ROW)["disposable_income"] == 2600.0
    assert "disposable_income" not in p.score(ROW), "the original must not see the emit"
    assert p.score_plan() is base_plan

    def flag(cap_by_income_band: float) -> bool:
        return cap_by_income_band < 60.0

    longer = p | flag
    assert longer.score_plan() is not base_plan
    assert longer.score(ROW)["flag"] is True
    assert "flag" not in p.score(ROW)

    routed = p.on_missing_input(default=Decision.DECLINE, reason=7)
    assert routed.score_plan() is not base_plan
    out = routed.score({**ROW, "net_income": None})
    assert (out["decision"], out["reason"], out["routed_on"]) == ("decline", 7, "net_income")
    assert p.score({**ROW, "net_income": None})["decision"] == "refer"

    bound = flow(disposable_income, affordability_ratio, module(cap_by_income_band).bind(cap=12.0))
    assert bound.score(ROW)["cap_by_income_band"] == 12.0
    assert p.score(ROW)["cap_by_income_band"] == 48.0


def test_a_hostile_in_place_field_swap_is_detected_not_served_stale():
    """The only way a frozen `Pipeline`'s fields can change after
    construction is `object.__setattr__`; the plan is keyed on the
    identity of the fields it was built from, so even that is caught."""

    def cap_by_income_band_v2(term_cap: float, min_net_salary: float) -> float:
        return term_cap * 0.5 if min_net_salary > 0 else term_cap

    p = _fresh_flagship()
    stale = p.score_plan()
    assert p.score(ROW)["cap_by_income_band"] == 48.0

    replacement = flow(disposable_income, affordability_ratio, cap_by_income_band_v2)
    object.__setattr__(p, "elements", replacement.elements)
    assert p.score_plan() is not stale
    out = p.score(ROW)
    assert out["cap_by_income_band_v2"] == 30.0 and "cap_by_income_band" not in out
    assert p.interface.terminals == replacement.interface.terminals

    object.__setattr__(p, "emits", p.emit("disposable_income").emits)
    assert p.score(ROW)["disposable_income"] == 2600.0

    from decider2.types import MissingInputPolicy

    object.__setattr__(p, "missing_input_policy", MissingInputPolicy(default=Decision.DECLINE, reason=99))
    assert p.score({**ROW, "expenses": None})["reason"] == 99


# --- 2. thread safety --------------------------------------------------------


def test_sixteen_threads_score_concurrently_with_no_cross_talk():
    """Serving runs concurrent `score()` on one pipeline (EXPERIMENTS.md
    §N4). Every thread scores its OWN distinct records, thousands of times,
    interleaved with fifteen others; a shared row buffer would show up as
    one thread's inputs answering another's record."""
    p = _fresh_flagship()
    p.score(ROW)  # warm, so no thread pays a compile
    n_threads, rounds = 16, 400
    pool_ids: set[int] = set()
    lock = threading.Lock()
    # Every thread reaches the loop before any starts it, so all sixteen
    # really are interleaving (an executor would let an early finisher
    # take a second task on the same thread, and the same pool).
    start = threading.Barrier(n_threads)
    results: list[int | None] = [None] * n_threads

    def worker(t: int) -> None:
        rows = [
            {
                "net_income": 3000.0 + 100.0 * t + 7.0 * k,
                "expenses": 500.0 + 3.0 * t + k,
                "instalment": 200.0 + t + k,
                "term_cap": 60.0 + t,
                "min_net_salary": (3000.0 + 100.0 * t) if k % 2 else 9000.0 + t,
            }
            for k in range(8)
        ]
        expected = [_expected(r) for r in rows]
        start.wait()
        mismatches = 0
        for _ in range(rounds):
            for r, e in zip(rows, expected):
                out = p.score(r)
                if out["affordability_ratio"] != e["affordability_ratio"] or out["cap_by_income_band"] != e["cap_by_income_band"]:
                    mismatches += 1
        with lock:
            pool_ids.add(id(p.score_plan()._pool()))
        results[t] = mismatches

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert results == [0] * n_threads, f"cross-talk between threads: {results}"
    assert len(pool_ids) == n_threads, "every thread must marshal into its own pooled row"


def test_a_re_entrant_call_on_one_thread_never_shares_the_busy_pool():
    plan = _fresh_flagship().score_plan()
    pool = plan._pool()
    assert plan._pool() is pool
    pool.busy = True
    try:
        assert plan._pool() is not pool
    finally:
        pool.busy = False
    assert plan._pool() is pool


def test_pooled_validity_and_fill_state_never_leaks_between_calls():
    """The same thread's pooled row is rewritten every call: an OPTIONAL
    input's validity flag and a MISSING_AS fill must reflect THIS record,
    whichever way the previous call left the buffer."""

    def net(net_income: float | None, expenses: float = missing_as(100.0)) -> float:
        return (-1.0 if net_income is None else net_income) - expenses

    p = flow(net)
    full = {"net_income": 10.0, "expenses": 3.0}
    assert p.score(full)["net"] == 7.0
    assert p.score({"net_income": None, "expenses": 3.0})["net"] == -4.0
    assert p.score(full)["net"] == 7.0
    assert p.score({"net_income": 10.0})["net"] == -90.0          # absent -> missing_as fill
    assert p.score({"net_income": 10.0, "expenses": None})["net"] == -90.0
    assert p.score(full)["net"] == 7.0
    assert p.score({})["net"] == -101.0                           # both absent: None, and the fill
    assert p.score(full)["net"] == 7.0

    # A REQUIRED input that is absent routes (doc 03 §1) before anything is
    # marshalled for it; the next full call on the same thread is unaffected.
    routed = flagship.score({**ROW, "instalment": None})
    assert routed["routed_on"] == "instalment" and "cap_by_income_band" not in routed
    assert flagship.score(ROW)["cap_by_income_band"] == 48.0


# --- 3. the retune contract (doc 08 §2) --------------------------------------


def test_retune_takes_effect_and_compiles_nothing():
    p = _fresh_flagship()
    p.precompile()
    assert p.score(ROW)["cap_by_income_band"] == 48.0
    with count_new_compiles() as counted:
        retuned = p.score(ROW, params={"cap_by_income_band": {"cap": 24.0}})
        back = p.score(ROW)
    assert retuned["cap_by_income_band"] == 24.0, "the new threshold must take effect"
    assert back["cap_by_income_band"] == 48.0, "and must not stick to the next, untuned call"
    assert counted.count == 0, f"a values-only retune compiled {counted.kinds!r}"


def test_serve_handle_generation_swap_takes_effect_and_compiles_nothing():
    handle = _fresh_flagship().serve()
    handle.warm()
    assert handle.score(ROW)["cap_by_income_band"] == 48.0
    handle.stage({"cap_by_income_band": {"cap": 24.0}})
    handle.activate()
    with count_new_compiles() as counted:
        out = handle.score(ROW)
    assert out["cap_by_income_band"] == 24.0
    assert counted.count == 0
    handle.rollback()
    assert handle.score(ROW)["cap_by_income_band"] == 48.0


def test_a_bad_retune_is_still_a_hard_error():
    p = _fresh_flagship()
    with pytest.raises(Exception, match="cap"):
        p.score(ROW, params={"cap_by_income_band": {"cap": 999.0}})   # le=60, pydantic
    with pytest.raises(ValueError, match="no module instance"):
        p.score(ROW, params={"cap_by_income_bnad": {"cap": 24.0}})


# --- 4. the ladder's fourth rung stays green ---------------------------------


def test_score_still_agrees_with_apply_on_the_flagship_and_an_optional_input():
    frame = pl.DataFrame({
        "net_income": [9200.0, 4100.0, 15000.0, 4999.0],
        "expenses": [3100.0, 1500.0, 6000.0, 2000.0],
        "instalment": [1200.0, 800.0, 2500.0, 700.0],
        "term_cap": [60.0, 60.0, 60.0, 60.0],
        "min_net_salary": [9200.0, 4100.0, 15000.0, 4999.0],
    })
    assert_equivalent(flagship, frame)
    assert_equivalent(flagship, frame, params={"cap_by_income_band": {"cap": 12.0}})

    def optional_net(net_income: float | None, expenses: float) -> float:
        return -expenses if net_income is None else net_income - expenses

    optional = flow(optional_net)
    assert_equivalent(optional, pl.DataFrame({"net_income": [1.0, None, 3.0], "expenses": [0.5, 0.5, 0.5]}))


# --- 5. the spec: 60 µs on the flagship ---------------------------------------


def test_flagship_score_p50_is_within_the_60us_spec():
    """Doc 02 §3.5's single-record budget, measured on the flagship: p50
    over 2000 calls with GC on. Before Stage 4 this was ~420 µs on the
    same box; the plan brings it to ~30 µs. Up to three rounds are taken
    so a transient load spike on a shared box cannot fail a real ~30 µs
    result, but a genuine regression past 60 µs still fails every round."""
    p = _fresh_flagship()
    p.precompile()
    p50s = []
    for _ in range(3):
        gc.collect()
        samples = []
        for _ in range(2000):
            t0 = time.perf_counter()
            p.score(ROW)
            samples.append(time.perf_counter() - t0)
        samples.sort()
        p50 = samples[len(samples) // 2] * 1e6
        p50s.append(p50)
        if p50 <= 60.0:
            return
    pytest.fail(f"flagship score() p50 over 60 µs in every round: {[round(x, 1) for x in p50s]} µs")
