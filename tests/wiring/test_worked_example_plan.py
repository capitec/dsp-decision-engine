"""The spec's worked example pipeline, resolved (a scalar step stands in for the tree)."""
from __future__ import annotations

from decider import branch, dag, flow, frame_step, missing_as, param, step
from decider.engine.wiring import Branch, Sequence, resolve


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def affordable(ratio: float, min_ratio: float = param(0.3, shared_key="min_ratio")) -> bool:
    return ratio >= min_ratio


def term_cap(requested_term: float, ceiling: float = param(60.0, ge=6, le=84)) -> float:
    return min(requested_term, ceiling)


@step(output="term_cap")
def cap_by_income(term_cap: float, min_net_salary: float = missing_as(0.0), cap: float = param(48.0)) -> float:
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


@step(name="risk_tree", output="risk_band")
def risk_tree(ratio: float, hi_thresh: float = param(0.7)) -> int:
    return int(ratio > hi_thresh)


@frame_step(reads=["client_id"], writes=["bureau_score"])
def join_bureau(df):
    return df


term = flow(
    term_cap,
    cap_by_income,
    branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by_sector"),
    name="term",
)
affordability = dag(disposable_income, ratio, affordable, name="affordability")
pipeline = (join_bureau | affordability | banding | term | risk_tree).emit("term_cap@*")


def test_input_columns_are_what_no_step_produces():
    plan = resolve(pipeline)
    assert [i.name for i in plan.inputs] == [
        "client_id", "net_income", "expenses", "instalment", "requested_term", "min_net_salary", "sector_code",
    ]


def test_outputs_are_inputs_final_values_and_every_term_cap_version():
    plan = resolve(pipeline)
    produced = {k: v.producer for k, v in plan.outputs.items() if v.producer is not None}
    assert produced == {
        "bureau_score": "join_bureau",
        "affordable": "affordability/affordable",
        "band": "banding",
        "band_score": "banding",
        "term_cap": "term/by_sector",
        "risk_band": "risk_tree",
        "term_cap@term/term_cap": "term/term_cap",
        "term_cap@term/cap_by_income": "term/cap_by_income",
        "term_cap@term/by_sector/cap_private": "term/by_sector/cap_private",
        "term_cap@term/by_sector/cap_public": "term/by_sector/cap_public",
        "term_cap@term/by_sector": "term/by_sector",
    }


def test_every_call_has_an_id_and_an_origin():
    plan = resolve(pipeline)
    assert [(c.id, c.node.origin.path) for c in plan.calls] == list(enumerate([
        "join_bureau", "affordability/disposable_income", "affordability/ratio", "affordability/affordable",
        "banding", "term/term_cap", "term/cap_by_income", "term/by_sector/is_private",
        "term/by_sector/cap_private", "term/by_sector/cap_public", "risk_tree",
    ]))
    assert all(c.node.origin.source for c in plan.calls)


def test_the_nested_structure_is_kept():
    plan = resolve(pipeline)
    kinds = [type(r).__name__ for r in plan.root.children]
    assert kinds == ["Call", "Sequence", "Call", "Sequence", "Call"]
    term_seq = plan.root.children[3]
    assert isinstance(term_seq, Sequence) and isinstance(term_seq.children[2], Branch)


def test_ratio_is_read_by_three_steps_from_one_version():
    plan = resolve(pipeline)
    by_path = {c.node.origin.path: c for c in plan.calls}
    made = by_path["affordability/ratio"].writes[0]
    for reader in ("affordability/affordable", "banding", "risk_tree"):
        assert by_path[reader].reads[0] is made
