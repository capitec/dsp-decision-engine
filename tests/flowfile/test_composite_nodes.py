"""Flow files build `dag`, `each` and `optimise` nodes like the Python constructors."""
import polars as pl

from decider import dag, each, flow, optimise
from decider.steps import _composites
from decider.steps.flowfile import _load
from decider.lab.testing import assert_equivalent
from tests.flowfile import components as comp
from tests.flowfile.test_load_flow import _write


def _load_doc(tmp_path, steps):
    doc = {"flow": "f", "components": "tests.flowfile.components", "steps": steps}
    return _load.load_flow(_write(tmp_path, "f.yaml", doc))


def test_dag_node_orders_by_dependency(tmp_path):
    loaded = _load_doc(tmp_path, [{"dag": "d", "steps": [{"step": "make_c"}, {"step": "make_b"}]}])
    hand = flow(dag(comp.make_c, comp.make_b, name="d"), name="f")
    frame = pl.DataFrame({"a": [1.0, 2.0]})
    assert assert_equivalent(loaded, frame)["c"].to_list() == [4.0, 6.0]
    assert assert_equivalent(hand, frame)["c"].to_list() == [4.0, 6.0]


def test_each_node_runs_the_item_flow_per_element(tmp_path):
    loaded = _load_doc(tmp_path, [{"each": "items", "column": "items", "steps": [{"step": "heavy"}]}])
    frame = pl.DataFrame({"items": [[{"weight": 5.0}, {"weight": 25.0}], []]})
    assert assert_equivalent(loaded, frame)["items"].to_list() == [
        [{"weight": 5.0, "heavy": False}, {"weight": 25.0, "heavy": True}], []]


def test_each_node_output_keeps_the_original_list(tmp_path):
    loaded = _load_doc(tmp_path, [{"each": "e", "column": "items", "output": "enriched",
                                   "steps": [{"step": "heavy"}]}])
    out = assert_equivalent(loaded, pl.DataFrame({"items": [[{"weight": 25.0}]]}))
    assert out["items"].to_list() == [[{"weight": 25.0}]]
    assert out["enriched"].to_list() == [[{"weight": 25.0, "heavy": True}]]


def test_optimise_node_keeps_the_best_candidate(tmp_path):
    loaded = _load_doc(tmp_path, [{"optimise": "best", "count": [{"step": "three"}],
                                   "evaluate": [{"step": "closeness"}], "max_candidates": 4}])
    hand = flow(optimise(comp.three, flow(comp.closeness, name="evaluate"), max_candidates=4, name="best"), name="f")
    out = assert_equivalent(loaded, pl.DataFrame([{}]))
    assert out["best_index"].to_list() == [2]
    assert out.equals(assert_equivalent(hand, pl.DataFrame([{}])))


def test_composite_nodes_take_suppress_checks(tmp_path):
    loaded = _load_doc(tmp_path, [{"dag": "d", "steps": [{"step": "make_b"}], "suppress_checks": ["x"]}])
    assert loaded.steps[0].suppress_checks == ("x",)


def test_every_composite_constructor_has_a_node_kind():
    assert set(_load._NODES) == {"step", *_composites.COMPOSITES}
