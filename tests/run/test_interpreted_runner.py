"""The interpreted runner: checkpoints per node, branches, loops and frame steps."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from decider import branch, flow, frame_step, loop, step
from decider.engine import Engine
from decider.engine.run import Checkpoint


def seed(requested_term: float) -> float:
    return requested_term


term_cap = step(seed, output="term_cap")


def is_private(sector_code: int) -> bool:
    return sector_code == 1


@step(output="term_cap")
def cap_private(term_cap: float) -> float:
    return min(term_cap, 54.0)


@step(output="term_cap")
def cap_public(term_cap: float) -> float:
    return min(term_cap, 60.0)


by_sector = flow(term_cap, branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by"))


def _checkpoints(pipeline, df):
    exe = Engine().bind(pipeline)
    state, params = exe.prepare(df)
    return [(c.when, c.origin.path) for c in exe.runner.iterate(exe.plan, state, params)]


def _frame(n):
    return pl.DataFrame({"requested_term": [72.0] * n, "sector_code": [1, 2] * (n // 2)})


def test_every_node_yields_before_and_after_once_whatever_the_row_count():
    assert _checkpoints(by_sector, _frame(2)) == [
        ("before", ""), ("before", "seed"), ("after", "seed"),
        ("before", "by"), ("before", "by/is_private"), ("after", "by/is_private"),
        ("before", "by/cap_private"), ("after", "by/cap_private"),
        ("before", "by/cap_public"), ("after", "by/cap_public"),
        ("after", "by"), ("after", ""),
    ]
    assert _checkpoints(by_sector, _frame(200)) == _checkpoints(by_sector, _frame(2))


def test_a_checkpoint_carries_the_node_origin():
    exe = Engine().bind(by_sector)
    state, params = exe.prepare(_frame(2))
    first = next(exe.runner.iterate(exe.plan, state, params))
    assert isinstance(first, Checkpoint) and first.origin is exe.plan.root.node.origin


def test_a_session_can_stop_before_a_node_and_inspect_after_it():
    exe = Engine().bind(by_sector)
    state, params = exe.prepare(_frame(2))
    steps = exe.runner.iterate(exe.plan, state, params)
    seed_call = exe.plan.calls[0]
    for c in steps:
        if (c.when, c.origin.path) == ("before", "seed"):
            assert seed_call.writes[0].id not in state.values
        if (c.when, c.origin.path) == ("after", "seed"):
            assert state.read(seed_call.writes[0])[0].tolist() == [72.0, 72.0]
            break


def test_an_arm_no_row_takes_yields_nothing():
    paths = {p for _, p in _checkpoints(by_sector, pl.DataFrame({"requested_term": [72.0], "sector_code": [1]}))}
    assert "by/cap_private" in paths and "by/cap_public" not in paths


def test_a_branch_merges_each_rows_arm():
    out = by_sector.run(pl.DataFrame({"requested_term": [72.0, 72.0, 50.0], "sector_code": [1, 2, 1]}))
    assert out["term_cap"].to_list() == [54.0, 60.0, 50.0]


def test_an_arm_leaving_a_modified_name_alone_keeps_the_prior_value():
    def keep(term_cap: float) -> float:
        return term_cap

    pipeline = flow(term_cap, branch(is_private, cap_private, keep, modifies=["term_cap"], name="by"))
    out = pipeline.run(pl.DataFrame({"requested_term": [72.0, 72.0], "sector_code": [1, 2]}))
    assert out["term_cap"].to_list() == [54.0, 72.0]


def band(score: float) -> int:
    return 0 if score < 500 else 1 if score < 700 else 2


@step(output="rate")
def low(score: float) -> float:
    return 0.2


@step(output="rate")
def mid(score: float) -> float:
    return 0.1


@step(output="rate")
def high(score: float) -> float:
    return 0.05


def test_an_int_condition_picks_the_arm_by_index():
    pipeline = branch(band, low, mid, high, modifies=["rate"], name="price")
    out = pipeline.run(pl.DataFrame({"score": [400.0, 800.0, 600.0]}))
    assert out["rate"].to_list() == [0.2, 0.05, 0.1]


def test_an_int_condition_out_of_range_is_an_error():
    pipeline = branch(band, low, mid, modifies=["rate"], name="price")
    with pytest.raises(ValueError, match="branch price: the condition picked arm 2 on 1 row"):
        pipeline.run(pl.DataFrame({"score": [400.0, 800.0]}))


def keep_going(best: float) -> bool:
    return best < 10


@step(output="best")
def improve(best: float, step_size: float) -> float:
    return best + step_size


FRAME = pl.DataFrame({"best": [0.0, 9.0, 20.0], "step_size": [3.0, 3.0, 1.0]})


def test_a_loop_runs_each_row_until_its_condition_fails():
    search = loop(keep_going, improve, carries=["best"], max_iterations=10, name="search")
    assert search.run(FRAME)["best"].to_list() == [12.0, 12.0, 20.0]


def test_a_loop_stops_at_max_iterations():
    search = loop(keep_going, improve, carries=["best"], max_iterations=2, name="search")
    assert search.run(FRAME)["best"].to_list() == [6.0, 12.0, 20.0]


def test_a_loop_body_yields_once_per_iteration_not_per_row():
    search = loop(keep_going, improve, carries=["best"], max_iterations=10, name="search")
    paths = [p for w, p in _checkpoints(search, FRAME) if w == "before"]
    assert paths.count("search/improve") == 4
    assert paths.count("search/keep_going") == 5


def test_a_loop_leaves_the_input_column_alone_until_it_ends():
    def doubled(best: float) -> float:
        return best * 2

    search = loop(keep_going, improve, carries=["best"], max_iterations=10, name="search")
    assert flow(search, doubled).run(FRAME)["doubled"].to_list() == [24.0, 24.0, 40.0]


def enrich(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(disposable_income=pl.col("net_income") - 100.0)


def ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


INCOME = pl.DataFrame({"net_income": [1100.0, 2100.0], "instalment": [100.0, 200.0], "client_id": [7, 8]})


def test_names_read_after_an_unknown_lineage_frame_come_from_what_it_returns():
    out = flow(frame_step(enrich), ratio).run(INCOME)
    assert out["ratio"].to_list() == [10.0, 10.0]
    assert out["client_id"].to_list() == [7, 8]
    assert out["disposable_income"].to_list() == [1000.0, 2000.0]


def test_a_frame_step_sees_values_computed_before_it():
    def doubled(net_income: float) -> float:
        return net_income * 2

    seen = {}

    def look(df: pl.DataFrame) -> pl.DataFrame:
        seen.update(df.to_dict(as_series=False))
        return df

    flow(doubled, frame_step(look)).run(INCOME)
    assert seen["doubled"] == [2200.0, 4200.0]


def test_a_barrier_missing_a_column_later_steps_read_is_an_error():
    with pytest.raises(ValueError, match="frame step drop_it returned no column 'instalment', which later steps read"):
        flow(frame_step(lambda df: enrich(df).drop("instalment"), name="drop_it"), ratio).run(INCOME)


def test_a_frame_step_missing_a_declared_column_is_an_error():
    lazy = frame_step(lambda df: df, name="lazy", reads=["client_id"], writes=["bureau_score"])
    with pytest.raises(ValueError, match="frame step lazy returned no column 'bureau_score', which it declares"):
        lazy.run(INCOME)


def test_a_frame_step_changing_the_row_count_is_an_error():
    with pytest.raises(ValueError, match="frame step head returned 1 rows for 2"):
        frame_step(lambda df: df.head(1), name="head").run(INCOME)


def test_a_frame_step_inside_an_arm_sees_only_that_arms_rows():
    def mark(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(term_cap=pl.col("term_cap") - pl.len().cast(pl.Float64))

    arm = frame_step(mark, name="mark", reads=["term_cap"], writes=["term_cap"])
    pipeline = flow(term_cap, branch(is_private, arm, cap_public, modifies=["term_cap"], name="by"))
    out = pipeline.run(pl.DataFrame({"requested_term": [72.0, 72.0, 72.0], "sector_code": [1, 2, 1]}))
    assert out["term_cap"].to_list() == [70.0, 60.0, 70.0]


def test_a_row_node_without_a_reference_is_called_with_row_params_and_consts():
    from decider.engine.ir.decls import Input, Output
    from decider.engine.ir.nodes import CallNode
    from decider.engine.ir.origin import Origin

    def kernel(row, params, consts):
        return (row[0] * consts[0],)

    node = CallNode(Origin("scale", "tests:kernel"), "row", kernel, (Input("x", float),), (Output("scaled", float),),
                    (), consts=(("factor", 3.0),))
    out = Engine().bind(node).run(pl.DataFrame({"x": [1.0, 2.0]}))
    assert out["scaled"].to_list() == [3.0, 6.0]


def test_state_keeps_every_version_of_a_waterfall():
    exe = Engine().bind(by_sector)
    state, params = exe.prepare(pl.DataFrame({"requested_term": [72.0, 72.0], "sector_code": [1, 2]}))
    for _ in exe.runner.iterate(exe.plan, state, params):
        pass
    assert [v.producer for v in state.versions("term_cap@*")] == ["seed", "by/cap_private", "by/cap_public", "by"]
    assert state.column("term_cap@by/cap_private").to_list() == [54.0, None]
    assert state.column("term_cap").to_list() == [54.0, 60.0]


def test_state_records_an_override_as_a_new_version():
    exe = Engine().bind(by_sector)
    state, _ = exe.prepare(pl.DataFrame({"requested_term": [72.0], "sector_code": [1]}))
    v = state.record("term_cap", "override@by", np.array([40.0]))
    assert state.versions("term_cap@override@by") == [v]
    assert state.versions("term_cap")[0] is v
    assert v.id == len(exe.plan.versions)
