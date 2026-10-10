import polars as pl

from decikit import branch, dag, each, flow, frame_step, loop, optimise, step
from decikit._engine import Engine, step_map, to_ir
from decikit._engine.compile import fingerprint
from decikit.steps import ConfigurableStep
from decikit.lab.testing import no_recompile



class Doubled(ConfigurableStep):
    column: str

    def to_ir(self, ctx):
        return ctx.call(self, _double, inputs={"x": self.column})


NAMES = ["int64_overflow", "DCD403"]


def _double(x: float) -> float:
    return x * 2.0


def _positive(x: float) -> bool:
    return x > 0.0


def _still(x: float) -> bool:
    return x < 0.0


def _origin(s):
    return to_ir(s).origin


def test_step_decorator_and_call_carry_suppressions_on_the_origin():
    @step(suppress_checks=NAMES)
    def capped(x: float) -> float:
        return x

    assert _origin(capped).suppress_checks == tuple(NAMES)
    assert _origin(step(_double, suppress_checks=NAMES)).suppress_checks == tuple(NAMES)
    assert _origin(step(_double)).suppress_checks == ()


def test_frame_step_carries_suppressions():
    @frame_step(suppress_checks=NAMES)
    def passthrough(df: pl.DataFrame) -> pl.DataFrame:
        return df

    assert _origin(passthrough).suppress_checks == tuple(NAMES)


def test_flow_and_dag_carry_suppressions():
    assert _origin(flow(_double, suppress_checks=NAMES)).suppress_checks == tuple(NAMES)
    assert _origin(dag(_double, suppress_checks=NAMES)).suppress_checks == tuple(NAMES)


def test_a_single_step_dag_with_suppressions_is_its_own_node():
    inner = step(_double)
    assert dag(inner) is inner
    assert dag(inner, suppress_checks=NAMES) is not inner


def test_branch_loop_each_and_optimise_carry_suppressions():
    arm = step(_double, output="y")
    b = branch(_positive, arm, step(_double, output="y", name="other"), modifies=["y"], name="b", suppress_checks=NAMES)
    assert _origin(b).suppress_checks == tuple(NAMES)

    count = step(lambda x: 1, name="n", output="count")
    body = step(_double, output="x")
    lp = loop(_positive, body, carries=["x"], max_iterations=3, name="lp", suppress_checks=NAMES)
    assert _origin(lp).suppress_checks == tuple(NAMES)

    e = each("items", flow(step(_double, name="d"), name="item"), name="items", suppress_checks=NAMES)
    assert _origin(e).suppress_checks == tuple(NAMES)

    @step(output="score")
    def score(index: int) -> float:
        return float(index)

    o = optimise(count, flow(score, name="evaluate"), max_candidates=4, name="best", suppress_checks=NAMES)
    assert _origin(o).suppress_checks == tuple(NAMES)


def test_a_composite_does_not_copy_suppressions_onto_its_members():
    inner = step(_double)
    outer = flow(inner, name="outer", suppress_checks=NAMES)
    assert step_map(outer)["outer/_double"] is inner
    assert to_ir(outer).children_[0].origin.suppress_checks == ()


def test_a_config_field_defaults_empty_and_round_trips_through_json():
    assert Doubled(name="t", column="c").suppress_checks == []
    cfg = Doubled(name="t", column="c", suppress_checks=NAMES)
    assert _origin(cfg).suppress_checks == tuple(NAMES)
    assert ConfigurableStep.load(cfg.model_dump_json()) == cfg
    assert cfg.model_dump()["suppress_checks"] == NAMES


def test_suppressions_survive_naming_a_copy():
    assert _origin(step(_double, suppress_checks=NAMES).named("again")).suppress_checks == tuple(NAMES)


def test_suppressions_stay_out_of_compile_keys_and_never_recompile():
    plain = flow(step(_double, name="d"), step(_still, name="s"), name="f")
    quiet = flow(step(_double, name="d", suppress_checks=NAMES), step(_still, name="s"), name="f",
                 suppress_checks=["DCD101"])
    assert [fingerprint(n.fn) for n in to_ir(plain).children_] == [fingerprint(n.fn) for n in to_ir(quiet).children_]
    row = {"x": 3.0}
    exe = Engine().bind(plain, "fused")
    exe.run(pl.DataFrame({"x": [1.0, 2.0]}))
    exe.run(row)
    with no_recompile():
        other = Engine().bind(quiet, "fused")
        other.run(pl.DataFrame({"x": [1.0, 2.0]}))
        assert other.run(row) == exe.run(row)
