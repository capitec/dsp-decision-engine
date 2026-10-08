"""A decision table (or tree) inside a loop or branch runs in the same kernel as its neighbours."""
import polars as pl

from decider import branch, flow, optimise, step
from decider._engine import Engine
from decider.lab.testing import assert_equivalent
from decider.steps.tables import DecisionTableConfig
from decider.steps.trees import TreeConfig

FRAME = pl.DataFrame({"n": [5, 40, 0]})


@step(output="count")
def count(n: int) -> int:
    return n


@step(output="grade")
def grade(index: int) -> float:
    return float(index % 4)


@step(output="score")
def score(index: int, pts: int) -> float:
    return pts * 10.0 - index


def table(output: str = "pts", default=0, rows=None) -> DecisionTableConfig:
    rows = rows or [{"g": float(g), "pts": g * 3 % 5, "tag": f"t{g}"} for g in range(4)]
    return DecisionTableConfig(
        name="lookup", columns={"g": "Float64", "pts": "Int64", "tag": "String"}, rows=rows,
        expression={"type": "eq", "variable": "grade", "value_column": "g"}, outputs=[output], default=[default])


def packed(pipeline) -> list[str]:
    exe = Engine().bind(pipeline, "fused")
    exe.run(FRAME)
    return sorted(exe.runner.packed)


def search(lookup) -> object:
    return flow(optimise(count, flow(grade, lookup, score, name="evaluate"), max_candidates=64, name="best"),
                name="pipeline")


def test_a_table_lookup_inside_optimise_is_packed_with_the_loop():
    pipeline = search(table())
    assert_equivalent(pipeline, FRAME)
    assert packed(pipeline) == ["pipeline/best/search"]


def test_a_table_with_a_null_output_inside_a_loop_agrees_in_every_mode():
    rows = [{"g": 0.0, "pts": None, "tag": "a"}, {"g": 1.0, "pts": 2, "tag": "b"}]

    @step(output="score")
    def nullable_score(index: int, pts: int | None) -> float:
        return 0.0 if pts is None else pts * 10.0 - index

    pipeline = flow(optimise(count, flow(grade, table(rows=rows, default=None), nullable_score, name="evaluate"),
                             max_candidates=64, name="best"), name="pipeline")
    assert_equivalent(pipeline, FRAME)


def test_a_table_lookup_in_a_branch_arm_agrees_in_every_mode():
    @step(output="big")
    def big(n: int) -> bool:
        return n > 10

    @step(output="out")
    def zero(n: int) -> int:
        return 0

    @step(output="out")
    def looked_up(pts: int) -> int:
        return pts

    @step(output="grade")
    def fixed(n: int) -> float:
        return float(n % 4)

    arm = flow(fixed, table(), looked_up, name="arm")
    pipeline = flow(branch(big, arm, zero, modifies=["out"], name="pick"), name="pipeline")
    assert_equivalent(pipeline, FRAME)
    assert packed(pipeline) == ["pipeline/pick"]


def test_a_tree_inside_optimise_is_packed_with_the_loop():
    tree = TreeConfig(name="lookup", tree={
        "nodes": [{"id": "root", "data": {"type": "unary", "condition": {"op": ">", "feature": "grade", "threshold": 1.0}}},
                  {"id": "hi", "data": {"type": "leaf", "result_idx": 0}},
                  {"id": "lo", "data": {"type": "leaf", "result_idx": 1}}],
        "edges": [{"source": "root", "target": "hi", "data": {"sourceIndex": 0}},
                  {"source": "root", "target": "lo", "data": {"sourceIndex": 1}}],
        "output": {"data": [{"pts": 4}, {"pts": 1}], "default": {"pts": 0}, "dtypes": [["pts", "Int64"]]}})
    pipeline = search(tree)
    assert_equivalent(pipeline, FRAME)
    assert packed(pipeline) == ["pipeline/best/search"]
