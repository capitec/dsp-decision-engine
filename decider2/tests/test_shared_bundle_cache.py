"""A `shared`-reading kernel compiled in one process is a genuine numba
cache HIT in the next one (doc 05 §4.2 condition 4, doc 08 §3.4's "editing
rows is free", doc 05 §8's caching story).

The defect this guards against: numba types a namedtuple argument by the
identity of its Python CLASS, and pickles that class into the on-disk
cache index. A class synthesised by `collections.namedtuple` inside a
function pickles BY VALUE (a dynamic class), so every fresh process
unpickled a different class, never matched, and recompiled every
`shared`-taking table kernel, forever — while writing a new, never-hit
index entry each time. `decider2.runtime.bundles` makes the class pickle
by reference to a deterministic, self-describing name instead, and
`Step.shared_fields` keeps a table step's bundle down to the keys it reads
so the class is the table's own, however many tables share `shared=`.

The deciding tests are the two-process ones: separate processes, a
persistent `NUMBA_CACHE_DIR`, `NUMBA_DEBUG_CACHE=1`, and the second
process must show `[cache] ... loaded` for the table's `row_fn`/`out_fn`
and zero `saved` for them.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numba.core.serialize import dumps, loads

import decider2
from decider2 import flow
from decider2.runtime import bundles
from decider2.runtime.bundles import _decode, _encode, bundle_class
from decider2.runtime.invoke import resolve_params
from decider2.tables import BetweenExpression, DecisionTable, ParametersConfig, table_module
from decider2.testing import assert_equivalent
from decider2.types import Input, Step

# ---------------------------------------------------------------------------
# The mechanism
# ---------------------------------------------------------------------------


def test_same_prefix_and_fields_is_the_same_class_object():
    a = bundle_class("_shared_params", ("base_rate", "t__between_lo"))
    b = bundle_class("_shared_params", ("base_rate", "t__between_lo"))
    assert a is b
    assert a._fields == ("base_rate", "t__between_lo")
    # Field ORDER is part of the identity: a bundle built from a dict with
    # the keys in another order is a different namedtuple, as before.
    assert bundle_class("_shared_params", ("t__between_lo", "base_rate")) is not a
    assert bundle_class("_other_params", ("base_rate", "t__between_lo")) is not a


def test_qualname_round_trips_prefix_and_every_field():
    # Double underscores, digits and a leading underscore on the prefix are
    # all things real keys contain (`bands__between_lo`, `_shared_params`),
    # and a length-prefixed encoding must not be confused by any of them.
    prefix = "_shared_params"
    fields = ("bands__n_rows", "b2__in_vals_start", "x_1_2_3", "_1_a")
    qualname = _encode(prefix, fields)
    assert qualname.isidentifier()
    assert "." not in qualname
    assert _decode(qualname) == (prefix, fields)


def test_class_is_registered_on_the_module_under_its_qualname():
    cls = bundle_class("_shared_params", ("base_rate",))
    assert cls.__module__ == bundles.__name__
    assert cls.__name__ == "_shared_params"  # readable numba type names
    assert getattr(bundles, cls.__qualname__) is cls


def test_class_pickles_by_reference_and_unpickles_to_the_same_object():
    cls = bundle_class("_shared_params", ("base_rate", "fee"))
    blob = dumps(cls)  # numba's own pickler, the one its cache index uses
    assert bundles.__name__.encode() in blob
    assert b"_make_skeleton_class" not in blob  # cloudpickle's by-value path
    assert loads(blob) is cls


def test_a_process_that_never_built_the_class_resolves_the_reference_to_its_own():
    """What a fresh process does when it loads an index another process
    wrote: `pickle.loads` asks the module for the qualname, the module's
    `__getattr__` builds the class, and that class is exactly what
    `bundle_class()` hands the kernel from then on."""
    key = ("_shared_params", ("only_here", "never_before"))
    cls = bundle_class(*key)
    blob = dumps(cls)
    # Simulate the fresh process: forget the class entirely.
    delattr(bundles, cls.__qualname__)
    del bundles._CLASSES[key]
    rebuilt = loads(blob)
    assert rebuilt is not cls
    assert rebuilt is bundle_class(*key)
    assert rebuilt._fields == key[1]


def test_unknown_module_attributes_still_raise_attribute_error():
    for name in ("nope", "Bundle", "Bundle_x", "Bundle_9_short", "__wrapped__"):
        with pytest.raises(AttributeError):
            getattr(bundles, name)
    assert not hasattr(bundles, "Bundle_3_ab")


def test_resolve_params_hands_the_kernel_a_registered_class():
    step = Step(
        name="uses_shared", fn=lambda x, shared: x * shared.base_rate,
        inputs=(Input("x", float),), params=(), reads_shared=True,
    )
    resolved = resolve_params([step], None, shared_overrides={"base_rate": 2.0, "fee": 1.5})
    cls = type(resolved.shared)
    assert cls is bundle_class("_shared_params", ("base_rate", "fee"))
    assert getattr(bundles, cls.__qualname__) is cls
    assert resolved.shared.base_rate == 2.0
    # A hand-written step declares no `shared_fields`: it gets the whole
    # bundle and nothing is projected for it.
    assert resolved.per_step_shared == {}


def test_shared_bundle_values_are_what_the_step_sees():
    # A sanity check that the registered class is a real namedtuple with
    # the values in field order — nothing about the registration changes
    # what a step receives.
    cls = bundle_class("_shared_params", ("a", "b"))
    inst = cls(a=np.array([1.0]), b=np.array([2.0]))
    assert tuple(inst) == (inst.a, inst.b)
    assert inst._asdict().keys() == {"a", "b"}


# ---------------------------------------------------------------------------
# Per-step projection: a table's bundle is its own, whatever else is merged
# ---------------------------------------------------------------------------


def _band_table(name: str, out: str) -> DecisionTable:
    return DecisionTable(
        name=name,
        parameters=ParametersConfig(
            data=[
                {"lo": None, "hi": 30.0, out: 1},
                {"lo": 30.0, "hi": 70.0, out: 2},
                {"lo": 70.0, "hi": None, out: 3},
            ],
            dtypes={"lo": "Float64", "hi": "Float64", out: "Int64"},
        ),
        expression=BetweenExpression(
            type="between", variable="score", lower_bound_column="lo", upper_bound_column="hi",
        ),
        outputs=[out],
        default=[0],
    )


def test_a_table_step_gets_only_the_fields_it_declares_and_the_same_class_alone_or_composed():
    a = table_module(_band_table("ta", "pa"))
    b = table_module(_band_table("tb", "pb"))
    row_a = a.encoded.row_step
    out_a = a.encoded.output_steps[0]
    assert row_a.shared_fields is not None and out_a.shared_fields is not None

    alone = resolve_params(a.module.steps, None, shared_overrides=a.shared)
    together = resolve_params(
        a.module.steps + b.module.steps, None, shared_overrides={**a.shared, **b.shared}
    )
    # The whole bundle grew (12 -> 24 fields) ...
    assert len(type(alone.shared)._fields) == 12
    assert len(type(together.shared)._fields) == 24
    # ... but the row step's OWN bundle did not change class, so neither
    # does its numba type, its compile cost or its cache entry.
    assert type(alone.per_step_shared[row_a.name]) is type(together.per_step_shared[row_a.name])
    assert type(alone.per_step_shared[row_a.name])._fields == row_a.shared_fields
    assert type(alone.per_step_shared[out_a.name])._fields == out_a.shared_fields
    assert len(out_a.shared_fields) == 2
    # Values are the table's own arrays, by name.
    np.testing.assert_array_equal(
        getattr(together.per_step_shared[row_a.name], "ta__between_lo"), a.shared["ta__between_lo"]
    )


def test_a_missing_declared_shared_field_is_a_clear_error_not_a_numba_typing_error():
    a = table_module(_band_table("ta", "pa"))
    incomplete = dict(a.shared)
    incomplete.pop("ta__between_lo")
    with pytest.raises(ValueError, match=r"ta_row.*ta__between_lo.*shared="):
        resolve_params(a.module.steps, None, shared_overrides=incomplete)


def test_two_composed_tables_agree_across_all_modes():
    a = table_module(_band_table("ta", "pa"))
    b = table_module(_band_table("tb", "pb"))
    frame = pl.DataFrame({"score": [10.0, 50.0, 90.0, 30.0, 70.0]})
    pipe = flow(a.module, b.module)
    assert_equivalent(pipe, frame, shared={**a.shared, **b.shared})
    out = pipe.apply(frame, shared={**a.shared, **b.shared})
    assert out["pa"].to_list() == out["pb"].to_list() == [1, 2, 3, 2, 3]


# ---------------------------------------------------------------------------
# The deciding tests: two processes
# ---------------------------------------------------------------------------

_CHILD = textwrap.dedent(
    """
    import contextlib, io, json, os
    import polars as pl
    from decider2 import flow
    from decider2.tables import BetweenExpression, DecisionTable, ParametersConfig, table_module

    def band_table(name, out):
        return DecisionTable(
            name=name,
            parameters=ParametersConfig(
                data=[
                    {"lo": None, "hi": 30.0, out: 1},
                    {"lo": 30.0, "hi": 70.0, out: 2},
                    {"lo": 70.0, "hi": None, out: 3},
                ],
                dtypes={"lo": "Float64", "hi": "Float64", out: "Int64"},
            ),
            expression=BetweenExpression(
                type="between", variable="score",
                lower_bound_column="lo", upper_bound_column="hi",
            ),
            outputs=[out], default=[0],
        )

    n = int(os.environ["D2_N_TABLES"])
    tables = [table_module(band_table(f"t{i}", f"p{i}")) for i in range(n)]
    shared = {}
    for t in tables:
        shared.update(t.shared)
    frame = pl.DataFrame({"score": [10.0, 50.0, 90.0, 30.0]})
    log = io.StringIO()
    with contextlib.redirect_stdout(log):  # NUMBA_DEBUG_CACHE prints to stdout
        out = flow(*[t.module for t in tables]).apply(frame, shared=shared)
    lines = log.getvalue().splitlines()
    print(json.dumps({
        "p0": out["p0"].to_list(),
        "saved": [l.rsplit("/", 1)[-1] for l in lines if "[cache] data saved" in l],
        "loaded": [l.rsplit("/", 1)[-1] for l in lines if "[cache] data loaded" in l],
    }))
    """
)


def _run_in_fresh_process(tmp_path: Path, n_tables: int = 1) -> dict:
    src_dir = str(Path(decider2.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src_dir, env.get("PYTHONPATH")) if p)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "numba_cache")
    env["NUMBA_DEBUG_CACHE"] = "1"
    env["D2_N_TABLES"] = str(n_tables)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.splitlines()[-1])


def _count(names: list[str], fn_name: str) -> int:
    return sum(1 for n in names if fn_name in n)


def test_a_shared_reading_table_is_a_genuine_cache_hit_in_a_fresh_process(tmp_path):
    cold = _run_in_fresh_process(tmp_path)
    warm = _run_in_fresh_process(tmp_path)

    assert cold["p0"] == warm["p0"] == [1, 2, 3, 2]
    # Cold: the shared-reading kernels were compiled and written out.
    assert _count(cold["saved"], "row_fn") == 1, cold
    assert _count(cold["saved"], "out_fn") == 1, cold
    # Warm, in a process that never saw the cold one's class objects: those
    # same entries are LOADED, and nothing at all is re-saved.
    assert warm["saved"] == [], warm
    assert _count(warm["loaded"], "row_fn") == 1, warm
    assert _count(warm["loaded"], "out_fn") == 1, warm


def test_a_table_compiled_alone_is_still_a_cache_hit_when_composed_with_another(tmp_path):
    """Adding a second table beside the first (`shared={**a, **b}`) must
    not invalidate the first's entries: its steps' bundles are projected
    to their own fields, so the bundle class — and with it the numba
    signature the cache is keyed on — is unchanged."""
    alone = _run_in_fresh_process(tmp_path, n_tables=1)
    composed = _run_in_fresh_process(tmp_path, n_tables=2)

    assert alone["p0"] == composed["p0"] == [1, 2, 3, 2]
    # t0's row_fn/out_fn: loaded. t1's: compiled for the first time (its
    # closure captures different keys, so it is a different entry).
    assert _count(composed["loaded"], "row_fn") == 1, composed
    assert _count(composed["loaded"], "out_fn") == 1, composed
    assert _count(composed["saved"], "row_fn") == 1, composed
    assert _count(composed["saved"], "out_fn") == 1, composed
