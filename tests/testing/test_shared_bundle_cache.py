"""Params bundles: one class per field set, pickled by reference, converted once per document and node."""
import json
import os
import subprocess
import sys
import textwrap

import polars as pl
import pytest
from numba.core.serialize import dumps, loads

import decider.engine.params.bundles as bundles
import decider.engine.params.validate as validate
from decider import flow, param
from decider.engine import Engine
from decider.engine.params import ParamsError, bundle_class
from decider.engine.params.bundles import _decode, _encode
from decider.testing import assert_equivalent

# --- the class ------------------------------------------------------------------


def test_field_order_is_part_of_the_class_identity():
    a = bundle_class(("base_rate", "cap"))
    assert bundle_class(("base_rate", "cap")) is a
    assert bundle_class(("cap", "base_rate")) is not a


def test_the_qualname_round_trips_awkward_field_names():
    fields = ("bands__n_rows", "b2__in_vals_start", "x_1_2_3", "_1_a")
    name = _encode(fields)
    assert name.isidentifier() and "." not in name
    assert _decode(name) == fields


def test_the_class_is_registered_on_the_module_and_pickles_by_reference_with_numbas_pickler():
    cls = bundle_class(("base_rate", "fee"))
    assert cls.__module__ == bundles.__name__
    assert getattr(bundles, cls.__qualname__) is cls
    blob = dumps(cls)
    assert b"_make_skeleton_class" not in blob  # cloudpickle's by-value path
    assert loads(blob) is cls


def test_a_process_that_never_built_the_class_rebuilds_it_from_the_reference():
    fields = ("only_here", "never_before")
    cls = bundle_class(fields)
    blob = dumps(cls)
    delattr(bundles, cls.__qualname__)
    del bundles._CLASSES[fields]
    rebuilt = loads(blob)
    assert rebuilt is not cls and rebuilt is bundle_class(fields)
    assert rebuilt._fields == fields


@pytest.mark.parametrize("name", ["nope", "Bundle_x", "Bundle_9_short", "Bundle_3_ab", "__wrapped__"])
def test_other_module_attributes_still_raise_attribute_error(name):
    with pytest.raises(AttributeError):
        getattr(bundles, name)


# --- per node, per document --------------------------------------------------------


def low_cap(x: float, rate: float = param(0.5, shared_key="rate", ge=0, le=1)) -> float:
    return x * rate


def high_cap(x: float, rate: float = param(2.0, shared_key="rate", ge=0, le=10),
             k: float = param(1.0)) -> float:
    return x * rate + k


FRAME = pl.DataFrame({"x": [1.0, 2.0, 3.0]})


def _bundle_type(pipeline, path):
    (node,) = [n for n in Engine().bind(pipeline).nodes.values() if n.path == path]
    return node.bundle_type


def test_a_nodes_bundle_holds_only_its_own_fields_whatever_it_is_composed_with():
    alone = _bundle_type(flow(low_cap), "low_cap")
    assert alone is _bundle_type(flow(low_cap, high_cap), "low_cap")
    assert alone is bundle_class(("rate",))
    assert _bundle_type(flow(low_cap, high_cap), "high_cap")._fields == ("rate", "k")


def test_a_shared_value_is_validated_against_each_nodes_own_bounds():
    exe = Engine().bind(flow(low_cap, high_cap))
    out = exe.run(FRAME, params={"shared": {"rate": 1.0}})
    assert out["high_cap"].to_list() == [2.0, 3.0, 4.0]
    with pytest.raises(ParamsError, match="low_cap: shared param 'rate'") as e:
        exe.run(FRAME, params={"shared": {"rate": 5.0}})
    assert "high_cap" not in str(e.value)


def test_a_bundle_is_converted_once_per_document_and_node(monkeypatch):
    seen = []
    original = validate.validate_node
    monkeypatch.setattr(validate, "validate_node", lambda node, doc: seen.append(node.path) or original(node, doc))
    exe = Engine().bind(flow(low_cap, high_cap), mode="fused")
    doc = {"shared": {"rate": 0.25}}
    exe.run(FRAME, params=doc)
    exe.score({"x": 1.0}, params={"shared": {"rate": 0.25}})
    exe.run(FRAME, params=doc)
    assert sorted(seen) == ["high_cap", "low_cap"]
    exe.run(FRAME, params={"shared": {"rate": 0.75}})
    assert sorted(seen) == ["high_cap", "high_cap", "low_cap", "low_cap"]


def test_two_nodes_sharing_a_key_agree_across_modes():
    out = assert_equivalent(flow(low_cap, high_cap), FRAME, params={"shared": {"rate": 0.5}, "high_cap": {"k": 2.0}})
    assert out["low_cap"].to_list() == [0.5, 1.0, 1.5]
    assert out["high_cap"].to_list() == [2.5, 3.0, 3.5]


# --- a params-reading kernel is a numba disk cache hit in a fresh process -----------

_CHILD = textwrap.dedent("""
    import contextlib, io, json
    import polars as pl
    from decider.engine import Engine
    from decider.engine.ir.decls import Input, Output, ParamDecl
    from decider.engine.ir.nodes import CallNode
    from decider.engine.ir.origin import Origin

    def band(row, params, consts):
        return (int(row[0] > params.cut),)

    node = CallNode(Origin("band", "tests:band"), "row", band, (Input("score", float),),
                    (Output("band", int),), (ParamDecl("cut", float, 50.0),))
    log = io.StringIO()
    with contextlib.redirect_stdout(log):  # NUMBA_DEBUG_CACHE prints to stdout
        out = Engine().bind(node, mode="stepped").run(pl.DataFrame({"score": [10.0, 90.0]}))
    lines = [l for l in log.getvalue().splitlines() if "band" in l]
    print(json.dumps({"band": out["band"].to_list(),
                      "saved": sum("data saved" in l for l in lines),
                      "loaded": sum("data loaded" in l for l in lines)}))
""")


def _run_in_fresh_process(tmp_path):
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba_cache"), NUMBA_DEBUG_CACHE="1")
    proc = subprocess.run([sys.executable, "child.py"], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=600, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.splitlines()[-1])


def test_a_params_reading_row_kernel_is_a_disk_cache_hit_in_a_fresh_process(tmp_path):
    (tmp_path / "child.py").write_text(_CHILD)  # once: numba's index is keyed by the file's mtime
    cold = _run_in_fresh_process(tmp_path)
    warm = _run_in_fresh_process(tmp_path)
    assert cold["band"] == warm["band"] == [0, 1]
    assert (cold["saved"], cold["loaded"]) == (1, 0), cold
    assert (warm["saved"], warm["loaded"]) == (0, 1), warm
