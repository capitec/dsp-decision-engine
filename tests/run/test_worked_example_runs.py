# The spec's worked example pipeline, run end to end in every mode.
# Trees arrive later, so `TreeConfig` is a stub emitting one row node with an
# njit-able kernel and a Python reference. No `from __future__ import annotations`: the stub's
# pydantic fields resolve against its (fake) module.
import polars as pl

from decider import branch, dag, flow, frame_step, missing_as, param, step
from decider.engine import Engine
from decider.engine.ir.decls import Input, Output, ParamDecl
from decider.engine.ir.nodes import CallNode
from decider.steps import ConfigurableStep, Value


class TreeConfig(ConfigurableStep):
    __module__ = "decider.steps.trees"
    threshold: Value[float] = 0.5

    def to_ir(self, ctx):
        threshold = ctx.value(self.threshold, float)
        params, consts = ((threshold,), ()) if isinstance(threshold, ParamDecl) else ((), (("threshold", threshold),))
        return CallNode(
            ctx.origin(self), "row", _tree_kernel, (Input("ratio", float),), (Output("risk_band", int),),
            params, reference=_tree_reference, consts=consts,
        )


def _tree_kernel(row, params, consts):
    return (int(row[0] > params[0]),)


def _tree_reference(row, params, consts, visit):
    visit("n0")
    return (int(row[0] > (params or consts)[0]),)


BUREAU = pl.DataFrame({"client_id": [1, 2], "bureau_score": [700, 650]})


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def affordable(ratio: float, min_ratio: float = param(0.3, shared_key="min_ratio")) -> bool:
    return ratio >= min_ratio


def term_cap(requested_term: float, ceiling: float = param(60.0, ge=6, le=84)) -> float:
    return min(requested_term, ceiling)


@step(output="term_cap")
def cap_by_income(term_cap: float, min_net_salary: float = missing_as(0.0),
                  cap: float = param(48.0, ge=6, le=60)) -> float:
    return min(term_cap, cap) if min_net_salary < 5000 else term_cap


@step(output="term_cap")
def cap_private(term_cap: float, cap: float = param(54.0)) -> float:
    return min(term_cap, cap)


@step(output="term_cap")
def cap_public(term_cap: float, cap: float = param(60.0)) -> float:
    return min(term_cap, cap)


def is_private(sector_code: int) -> bool:
    return sector_code == 1


@step(outputs=("band", "band_score"))
def banding(ratio: float) -> tuple[int, float]:
    return (1, 10.0) if ratio > 2 else (0, 0.0)


@frame_step(reads=["client_id"], writes=["bureau_score"])
def join_bureau(df: pl.DataFrame) -> pl.DataFrame:
    return df.join(BUREAU, on="client_id", how="left")


affordability = dag(disposable_income, ratio, affordable, name="affordability")
term = flow(
    term_cap,
    cap_by_income,
    branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by_sector"),
    name="term",
)
risk_tree = TreeConfig.model_validate_json(
    '{"type": "decider.steps.trees:TreeConfig", "name": "risk_tree",'
    ' "threshold": {"param": "hi_thresh", "default": 0.7}}'
)
pipeline = (join_bureau | affordability | banding | term | risk_tree).emit("term_cap@*")

FRAME = pl.DataFrame({
    "client_id": [1, 2, 3],
    "net_income": [9200.0, 4100.0, 15000.0],
    "expenses": [3100.0, 3700.0, 6000.0],
    "instalment": [1200.0, 800.0, 5000.0],
    "requested_term": [72.0, 50.0, 84.0],
    "min_net_salary": [9200.0, None, 15000.0],
    "sector_code": [1, 2, 1],
})


def test_the_worked_example_runs_end_to_end(bind):
    out = bind(pipeline).run(FRAME)
    assert out.columns == [
        *FRAME.columns, "bureau_score", "affordable", "band", "band_score", "term_cap", "risk_band",
        "term_cap@term/term_cap", "term_cap@term/cap_by_income", "term_cap@term/by_sector/cap_private",
        "term_cap@term/by_sector/cap_public", "term_cap@term/by_sector",
    ]
    assert out["bureau_score"].to_list() == [700, 650, None]
    assert out["affordable"].to_list() == [True, True, True]
    assert out["band"].to_list() == [1, 0, 0]
    assert out["band"].dtype == pl.Int64
    assert out["band_score"].to_list() == [10.0, 0.0, 0.0]
    assert out["risk_band"].to_list() == [1, 0, 1]
    assert out["term_cap"].to_list() == [54.0, 48.0, 54.0]
    assert out["term_cap@term/term_cap"].to_list() == [60.0, 50.0, 60.0]
    assert out["term_cap@term/cap_by_income"].to_list() == [60.0, 48.0, 60.0]
    assert out["term_cap@term/by_sector/cap_private"].to_list() == [54.0, None, 54.0]
    assert out["term_cap@term/by_sector/cap_public"].to_list() == [None, 48.0, None]
    assert out["term_cap@term/by_sector"].to_list() == [54.0, 48.0, 54.0]


def test_the_worked_example_retunes_arms_shared_params_and_the_tree(bind):
    params = {
        "shared": {"min_ratio": 1.0},
        "term": {"by_sector": {"cap_private": {"cap": 50.0}}},
        "risk_tree": {"hi_thresh": 2.0},
    }
    out = bind(pipeline).run(FRAME, params=params)
    assert out["affordable"].to_list() == [True, False, True]
    assert out["term_cap"].to_list() == [50.0, 48.0, 50.0]
    assert out["risk_band"].to_list() == [1, 0, 0]


def test_the_tree_reference_reports_its_internal_nodes():
    exe = Engine().bind(pipeline)
    visited = []
    exe.runner.visit = visited.append
    exe.run(FRAME)
    assert visited == ["n0", "n0", "n0"]


def test_score_runs_the_worked_example_for_one_record(bind):
    out = bind(pipeline).score(FRAME.row(1, named=True))
    assert (out["term_cap"], out["bureau_score"], out["risk_band"]) == (48.0, 650, 0)


def test_all_three_modes_agree_on_the_worked_example():
    params = {"shared": {"min_ratio": 1.0}, "risk_tree": {"hi_thresh": 2.0}}
    runs = [Engine().bind(pipeline, mode=m).run(FRAME, params=params) for m in ("interpreted", "stepped", "fused")]
    assert runs[0].equals(runs[1]) and runs[1].equals(runs[2])
    assert runs[0].schema == runs[1].schema == runs[2].schema


def test_score_equals_run_row_for_row_on_the_worked_example(bind):
    exe = bind(pipeline)
    batch = exe.run(FRAME).rows(named=True)
    assert [exe.score(row) for row in FRAME.iter_rows(named=True)] == batch
