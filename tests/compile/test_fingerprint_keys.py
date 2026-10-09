"""Fingerprints key classes, enum members, dates and plain containers by value, so they hold across processes."""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction

import pytest

from decider import flow
from decider._engine.compile import fingerprint
from decider.lab import diff
from decider.ops.decimal import Rounding

_STEPS = '''
import datetime, functools, re
from decimal import Decimal
from decider.ops import decimal
from decider.ops.decimal import Rounding

LIMIT = datetime.date(2024, 1, 1)
CODES = {"gold": (1, 2), "tiers": [3, 4], "fresh": frozenset({"a", "b", "c", "d", "e"})}
PATTERN = re.compile("^a+$")
HALF = Decimal("0.5")


@functools.lru_cache(maxsize=None)
def bump(x):
    return x + 1


def rounded(x: float) -> float:
    return decimal.round(x, 2, Rounding.HALF_UP)


def dated(x: float) -> float:
    return x + LIMIT.year


def coded(x: float) -> float:
    return x + len(CODES["tiers"]) + float(HALF) + bump(1) + (1 if PATTERN.match("aa") else 0)


def classed(x: float) -> float:
    return x if isinstance(x, (Decimal, Rounding)) else 0.0
'''

_CHILD = """
import json
import steps
from decider._engine.compile import fingerprint

names = ("rounded", "dated", "coded", "classed")
print(json.dumps({n: fingerprint(getattr(steps, n), strict=True) for n in names}))
"""


def test_the_key_of_a_step_reading_classes_dates_and_containers_is_the_same_in_every_process(tmp_path):
    (tmp_path / "steps.py").write_text(_STEPS)
    (tmp_path / "child.py").write_text(_CHILD)

    def keys(seed):
        proc = subprocess.run([sys.executable, "child.py"], cwd=tmp_path, env=dict(os.environ, PYTHONHASHSEED=seed),
                              capture_output=True, text=True, timeout=600, check=False)
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.splitlines()[-1])

    assert keys("1") == keys("2")


_CACHED = "from decider.ops import decimal\nfrom decider.ops.decimal import Rounding\n\n\n" \
          "def r(x: float) -> float:\n    return decimal.round(x, 2, Rounding.HALF_UP)\n"
_CACHE_CHILD = """
import json
import step
import decider._engine.compile.njit as njit
_, kernel = njit.jit(step.r)
print(json.dumps([kernel(3.14159), len(kernel.stats.cache_hits), len(kernel.stats.cache_misses)]))
"""


def test_a_step_using_decimal_round_hits_the_disk_cache_in_the_next_process(tmp_path):
    (tmp_path / "step.py").write_text(_CACHED)
    (tmp_path / "child.py").write_text(_CACHE_CHILD)
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba_cache"))

    def run():
        proc = subprocess.run([sys.executable, "child.py"], cwd=tmp_path, env=env, capture_output=True, text=True,
                              timeout=600, check=False)
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.splitlines()[-1])

    assert run() == [3.14, 0, 1]
    assert run() == [3.14, 1, 0]


def rounded(x: float) -> float:
    from decider.ops import decimal

    return decimal.round(x, 2, Rounding.HALF_UP)


_PICK = "from shapes import P\n\n\ndef pick(x: float) -> float:\n    return P(x, 1.0).b\n"
_PICK_CHILD = """
import json
import step
import decider._engine.compile.njit as njit
_, kernel = njit.jit(step.pick)
print(json.dumps(kernel(7.0)))
"""


def test_a_namedtuple_edited_in_another_module_is_not_served_from_the_disk_cache(tmp_path):
    (tmp_path / "step.py").write_text(_PICK)
    (tmp_path / "child.py").write_text(_PICK_CHILD)
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba_cache"))

    def run(fields):
        (tmp_path / "shapes.py").write_text(f"from collections import namedtuple\nP = namedtuple('P', '{fields}')\n")
        proc = subprocess.run([sys.executable, "child.py"], cwd=tmp_path, env=env, capture_output=True, text=True,
                              timeout=600, check=False)
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.splitlines()[-1])

    assert run("a b") == 1.0
    assert run("b a") == 7.0


def test_a_pipeline_calling_decimal_round_has_no_diff_against_itself():
    assert diff.structure(flow(rounded, name="t"), flow(rounded, name="t")).identical


def make(value):
    def rule(x: float) -> float:
        return x * len(repr(value))

    return rule


def key(value):
    return fingerprint(make(value), strict=True)


def test_a_date_constant_changes_the_key():
    assert key(datetime.date(2024, 1, 1)) == key(datetime.date(2024, 1, 1))
    assert key(datetime.date(2024, 1, 1)) != key(datetime.date(2024, 1, 2))


def test_a_class_is_keyed_by_module_and_qualname():
    def named(module, qualname):
        cls = type("K", (), {"__module__": module})
        cls.__qualname__ = qualname
        return cls

    assert key(named("a", "K")) == key(named("a", "K")) != key(named("b", "K"))
    assert key(named("a", "K")) != key(named("a", "Outer.K"))


def test_an_edited_class_body_changes_the_key():
    import collections

    def body(**attrs):
        return type("K", (), {"__module__": "a", **attrs})

    assert key(body(CAP=5.0)) == key(body(CAP=5.0)) != key(body(CAP=10.0))
    assert key(body(f=lambda self: 1)) != key(body(f=lambda self: 2))
    assert key(body(f=staticmethod(lambda: 1))) != key(body(f=staticmethod(lambda: 2)))
    assert key(collections.namedtuple("P", "a b")) != key(collections.namedtuple("P", "b a"))


def test_an_enum_member_is_keyed_by_class_and_name_and_its_class_by_its_members():
    import enum

    def make_enum(value):
        return enum.Enum("Mode", {"UP": value, "DOWN": 2})

    assert key(Rounding.HALF_UP) != key(Rounding.HALF_EVEN)
    assert key(make_enum(1)) == key(make_enum(1)) != key(make_enum(3))
    assert key(make_enum(1).UP) != key(make_enum(1).DOWN)


DISTINCT = [
    (1, "1"), (1, True), (1, 1.0), ((1, (2,)), ((1,), 2)), ([1], (1,)), ({1: 2}, {2: 1}), ({1, 2}, frozenset({1, 2})),
    ({"a": 1, "b": 2}, {"b": 2, "a": 1}), (Decimal("1.5"), Decimal("1.50")), (Decimal("1.5"), 1.5), (Fraction(1, 3), 0.5),
    (datetime.date(2024, 1, 1), datetime.datetime(2024, 1, 1)), (datetime.timedelta(1), datetime.timedelta(2)),
    ({"a": [1, 2]}, {"a": [1], "b": [2]}), (("ab", "c"), ("a", "bc")), (Decimal, Fraction),
]


@pytest.mark.parametrize("a, b", DISTINCT, ids=[f"{a!r}-{b!r}"[:40] for a, b in DISTINCT])
def test_values_that_differ_have_different_keys(a, b):
    assert key(a) != key(b)


def test_a_set_has_the_same_key_however_it_was_built():
    assert key({"gold", "silver", "bronze"}) == key({"bronze", "silver", "gold"})
    assert key(frozenset(range(50))) == key(frozenset(reversed(range(50))))


def test_a_container_that_holds_itself_does_not_loop():
    loop = []
    loop.append(loop)
    with pytest.raises(ValueError, match="mutable or arbitrary"):
        key(loop)


def test_an_arbitrary_object_is_refused_in_strict_mode_with_its_type_named():
    class Register:
        pass

    with pytest.raises(ValueError, match=r"Register object has no content key"):
        key(Register())
    assert fingerprint(make(Register()))  # non-strict still keys it by identity


def test_a_counter_global_keeps_its_key_as_it_fills():
    import collections

    calls = collections.Counter()

    def rule(x: float) -> float:
        calls["n"] += 1
        return x

    first = fingerprint(rule)
    rule(1.0)
    assert fingerprint(rule) == first
