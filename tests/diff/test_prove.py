"""prove: identical only from equal IR content hashes; everything else is "can't tell"."""
import numpy as np

from decider import ConfigurableStep, ParamTable, Value, flow, param_table
from decider.lab import diff
from decider.steps.trees import TreeConfig


def net(a: float, b: float) -> float:
    return a - b


def scaled(net: float) -> float:
    return net * 2.0


def scaled_changed(net: float) -> float:
    return net * 3.0


PIPELINE = flow(net, scaled, name="term")
CHANGED = flow(net, scaled_changed, name="term")


def _threshold_tree(threshold):
    rule = {"meta": {"name": "r"}, "rule": {"type": "unary", "then": {"id": "hit", "type": "leaf", "result_idx": 0},
            "condition": {"op": ">", "feature": "amount", "threshold": threshold}}}
    return TreeConfig.load({"type": "tree", "name": "t", "path_output": "leaf", "tree": {
        "type": "prioritized_flat_rule", "mode": "all", "rules": [rule],
        "output": {"data": [{"hit": 1}], "default": {"hit": 0}, "dtypes": [["hit", "Int64"]]}}})


class _Banded(ConfigurableStep):
    column: str
    factor: Value[float]
    bump: int = -1

    def to_ir(self, ctx):
        table = np.zeros(2000)
        table[self.bump] = 1.0
        return ctx.call(self, _banded, inputs={"x": self.column}, values={"factor": self.factor, "table": table})


def _banded(x: float, factor: float, table: object) -> float:
    return x * factor


def _ladder(rate_at_25):
    rows = [{"floor": i, "rate": rate_at_25 if i == 25 else 0.2} for i in range(50)]

    def rate(score: int, ladder: ParamTable = param_table({"floor": int, "rate": float}, default=rows)) -> float:
        return 0.0
    return flow(rate, name="pricing")


def test_the_same_pipeline_is_proven_identical():
    proof = diff.prove(PIPELINE, PIPELINE)
    assert (proof.tier, proof.identical, proof.evidence) == ("ir_hash", True, None)


def test_a_pipeline_rebuilt_from_the_same_code_is_proven_identical():
    assert diff.prove(PIPELINE, flow(net, scaled, name="term")).identical is True


def test_a_renamed_pipeline_with_the_same_code_is_still_identical():
    assert diff.prove(PIPELINE, flow(net, scaled, name="other")).identical is True


def test_changed_code_is_not_proven_and_carries_the_structure_as_evidence():
    proof = diff.prove(PIPELINE, CHANGED)
    assert proof.identical is None and proof.tier == "none"
    assert [c.ref.path for c in proof.evidence.changed_steps] == ["term/scaled_changed"] or proof.evidence.removed


def test_a_differing_threshold_gives_a_step_level_counterexample_not_a_verdict():
    proof = diff.prove(_threshold_tree(5.0), _threshold_tree(6.0))
    assert (proof.tier, proof.identical) == ("smt", None)
    assert proof.evidence.counterexample is not None and "step alone" in proof.reason


def test_equal_constants_are_proven_identical():
    assert diff.prove(_threshold_tree(5.0), _threshold_tree(5.0)).identical is True


def test_a_differing_param_default_is_not_proven_identical():
    low, high = ({"param": "limit", "default": d} for d in (5.0, 6.0))
    assert diff.prove(_threshold_tree(low), _threshold_tree(high)).identical is None


def test_a_large_constant_differing_in_an_elided_element_is_not_proven_identical():
    one, other = (_Banded(name="s", column="x", factor=2.0, bump=b) for b in (500, 1000))
    assert diff.prove(one, other).identical is None


def test_a_param_table_differing_in_one_middle_row_is_not_proven_identical():
    assert diff.prove(_ladder(0.1), _ladder(0.3)).identical is None


def test_equal_param_tables_are_proven_identical():
    assert diff.prove(_ladder(0.1), _ladder(0.1)).identical is True


def test_a_constant_that_cannot_be_hashed_by_content_is_never_proven_identical():
    class Opaque:
        pass

    class _Holds(ConfigurableStep):
        column: str

        def to_ir(self, ctx):
            return ctx.call(self, _banded, inputs={"x": self.column}, values={"factor": 1.0, "table": Opaque()})

    one, other = _Holds(name="s", column="x"), _Holds(name="s", column="x")
    assert diff.prove(one, other).identical is None


class _Opaque:
    pass


_THING = _Opaque()


def _uses_thing(x: float) -> float:
    return x if _THING else 0.0


def test_a_hash_never_reads_an_identity_keyed_global_as_equal():
    from decider.lab.diff._hash import snapshot

    assert snapshot(flow(_uses_thing, name="g")).hash != snapshot(flow(_uses_thing, name="g")).hash
