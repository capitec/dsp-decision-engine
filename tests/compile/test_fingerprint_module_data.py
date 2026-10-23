"""A step that reads `mod.NAME` depends on that value: editing it changes the key, in this process and the next."""
import enum
import json
import os
import subprocess
import sys
import types

import pytest

from decikit import flow, step
from decikit._engine.compile import fingerprint
from decikit.lab import diff


class Tier(enum.Enum):
    GOLD = 1


def reader(**attrs):
    module = types.ModuleType("limits")
    module.__dict__.update(attrs)
    namespace = {"limits": module}
    exec("def rule(x: float) -> float:\n    return x * limits.K + len(limits.NAMES) + limits.TIER.value\n", namespace)
    return namespace["rule"]


def test_editing_a_module_constant_changes_the_key():
    base = dict(K=5.0, NAMES=("a",), TIER=Tier.GOLD)
    assert fingerprint(reader(**base)) == fingerprint(reader(**base))
    assert fingerprint(reader(**base)) != fingerprint(reader(**{**base, "K": 6.0}))
    assert fingerprint(reader(**base)) != fingerprint(reader(**{**base, "NAMES": ("a", "b")}))


def test_a_module_dict_constant_is_keyed_by_value():
    low, high = (reader(K=1.0, NAMES={"a": 1}, TIER=Tier.GOLD), reader(K=1.0, NAMES={"a": 2}, TIER=Tier.GOLD))
    assert fingerprint(low) != fingerprint(high)


def test_a_module_attribute_that_cannot_be_read_does_not_break_the_walk():
    class Hostile(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError("no such luck")

    module = Hostile("limits")
    module.K, module.NAMES, module.TIER = 1.0, (), Tier.GOLD
    namespace = {"limits": module}
    exec("def rule(x: float) -> float:\n    return x * limits.K + limits.missing\n", namespace)
    assert fingerprint(namespace["rule"])


def test_a_structure_diff_sees_an_edited_module_constant():
    def pipeline(k):
        return flow(step(reader(K=k, NAMES=(), TIER=Tier.GOLD)), name="p")

    assert diff.structure(pipeline(5.0), pipeline(5.0)).identical
    assert not diff.structure(pipeline(5.0), pipeline(6.0)).identical


_CHILD = """
import json, sys
import steps
from decikit import flow
from decikit._engine.compile import fingerprint

exe = flow(steps.via_attr, name="p").compile("fused")
print(json.dumps([fingerprint(steps.via_attr, strict=True), exe.run({"x": 1.0})["via_attr"], exe.fallbacks()]))
"""

_STEPS = "import limits\n\n\ndef via_attr(x: float) -> float:\n    return x * limits.K + limits.PAIR[1]\n"


def test_an_edited_module_constant_is_not_served_from_the_previous_process_disk_cache(tmp_path):
    (tmp_path / "steps.py").write_text(_STEPS)
    (tmp_path / "child.py").write_text(_CHILD)

    def run(k, seed):
        (tmp_path / "limits.py").write_text(f"K = {k}\nPAIR = (1, 2)\n")
        # No .pyc: limits.py is rewritten at the same size within a second, which a .pyc cannot tell apart.
        env = dict(os.environ, PYTHONHASHSEED=seed, NUMBA_CACHE_DIR=str(tmp_path / "numba_cache"),
                   PYTHONPATH=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([sys.executable, "child.py"], cwd=tmp_path, env=env, capture_output=True, text=True,
                              timeout=600, check=False)
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.splitlines()[-1])

    first, same, edited = run(5.0, "1"), run(5.0, "2"), run(6.0, "3")
    assert first[1:3] == [7.0, {}]  # the constant compiles into the kernel
    assert first[0] == same[0] != edited[0]
    assert edited[1] == 8.0
