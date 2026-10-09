"""A step calling the standard library or an installed package is keyed by that function's name and version, not walked."""
import contextlib
import importlib
import json
import os
import subprocess
import sys

from decider._engine.compile import fingerprint
from decider.ops import decimal

fp = importlib.import_module("decider._engine.compile.fingerprint")

_STEPS = '''
import dataclasses, json, statistics
import polars as pl


def calls_libraries(x: float) -> float:
    return statistics.mean([x, 1.0]) + len(json.dumps({"a": x})) + float(dataclasses.is_dataclass(x)) + float(pl.lit(1) is None)
'''

_CHILD = "import json, steps\nfrom decider._engine.compile import fingerprint\n" \
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


def test_a_library_function_is_keyed_by_name_and_package_version(monkeypatch):
    import statistics

    first = fingerprint(statistics.mean)
    assert first != fingerprint(statistics.median)
    monkeypatch.setattr(fp, "_version", lambda module: "99.0")
    assert fingerprint(statistics.mean) != first


def test_deciders_own_functions_are_still_walked():
    assert not fp._is_library(decimal.round.__code__.co_filename)
    assert not fp._is_library(__file__)


def wrapped(value):
    @contextlib.contextmanager
    def step():
        yield value

    return step


def test_a_library_wrapper_still_keys_the_step_function_it_closes_over():
    assert fingerprint(wrapped(1)) == fingerprint(wrapped(1)) != fingerprint(wrapped(2))
