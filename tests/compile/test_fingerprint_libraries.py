"""A step calling the standard library or an installed package is keyed by that function's name and version, not walked."""
import contextlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys

import pytest

from decikit import flow, step
from decikit._engine.compile import fingerprint, hidden_state
from decikit.ops import decimal

fp = importlib.import_module("decikit._engine.compile.fingerprint")

_STEPS = '''
import dataclasses, json, statistics
import polars as pl


def calls_libraries(x: float) -> float:
    return statistics.mean([x, 1.0]) + len(json.dumps({"a": x})) + float(dataclasses.is_dataclass(x)) + float(pl.lit(1) is None)
'''

_CHILD = "import json, steps\nfrom decikit._engine.compile import fingerprint\n" \
         "print(json.dumps(fingerprint(steps.calls_libraries, strict=True)))\n"


def test_a_step_calling_library_functions_has_a_strict_key_that_holds_across_processes(tmp_path):
    (tmp_path / "steps.py").write_text(_STEPS)
    (tmp_path / "child.py").write_text(_CHILD)

    def key(seed):
        proc = subprocess.run([sys.executable, "child.py"], cwd=tmp_path, env=dict(os.environ, PYTHONHASHSEED=seed),
                              capture_output=True, text=True, timeout=600, check=False)
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout)

    assert key("1") == key("2")


def calling(fn):
    def step_(x: float) -> float:
        return fn(x)

    return step_


def test_a_library_function_a_step_calls_is_keyed_by_name_and_package_version(monkeypatch):
    import statistics

    first = fingerprint(calling(statistics.mean))
    assert first != fingerprint(calling(statistics.median))
    monkeypatch.setattr(fp, "_version", lambda module: "99.0")
    assert fingerprint(calling(statistics.mean)) != first


def test_the_toolkits_own_functions_are_still_walked():
    assert not fp._is_library(decimal.round.__code__.co_filename)
    assert not fp._is_library(__file__)


def wrapped(value):
    @contextlib.contextmanager
    def step():
        yield value

    return step


def test_a_library_wrapper_still_keys_the_step_function_it_closes_over():
    assert fingerprint(calling(wrapped(1))) == fingerprint(calling(wrapped(1))) != fingerprint(calling(wrapped(2)))


_INSTALLED = '''
class Register:
    cap = 3.0


REGISTER = Register()


def make_cap(limit):
    def cap(x: float) -> float:
        return min(x, limit)

    return cap


low = lambda x: min(x, 1.0)  # noqa: E731
high = lambda x: min(x, 2.0)  # noqa: E731


def reads_register(x: float) -> float:
    return x + REGISTER.cap
'''


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """A module of a package installed where a wheel puts it, beside the standard library."""
    (tmp_path / "shipped_steps.py").write_text(_INSTALLED)
    monkeypatch.setattr(fp, "_LIBRARY_ROOTS", (*fp._LIBRARY_ROOTS, str(tmp_path) + os.sep))
    fp._is_library.cache_clear()
    spec = importlib.util.spec_from_file_location("shipped_steps", tmp_path / "shipped_steps.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    fp._is_library.cache_clear()


def test_two_closures_from_one_installed_factory_are_different_steps(installed):
    low, high = installed.make_cap(10.0), installed.make_cap(20.0)
    assert fingerprint(calling(low)) != fingerprint(calling(high))
    pipeline = flow(step(low, output="low").named("low"), step(high, output="high").named("high"), name="p")
    for mode in ("interpreted", "stepped", "fused"):
        out = pipeline.compile(mode).score({"x": 50.0})
        assert (out["low"], out["high"]) == (10.0, 20.0), mode


def test_two_installed_lambdas_are_different_functions(installed):
    assert fingerprint(calling(installed.low)) != fingerprint(calling(installed.high))


def test_a_step_from_an_installed_package_is_still_checked_for_hidden_state(installed):
    assert [h[:3] for h in hidden_state(installed.reads_register)] == [("REGISTER", "object", "Register")]
