"""
Regression test for decider2 doc 05 section 9, acceptance criterion 8:

    "A regression test asserts per-call dispatch stays near 1 us, guarding the
     same-name-NamedTuple trap (doc 01 4c) that silently costs 15-24 us forever."

WHAT THE MEASUREMENT FOUND (see experiment.py / README.md):

  On numba 0.67.0 / python 3.14.5 the trap does NOT cost 15-24 us. It costs
  1.03x -- i.e. nothing. A timing assertion therefore has ZERO power to catch it
  on this stack, and a test suite that relies on one would ship the bug green.

  What the collision DOES do on this stack is silently substitute the type:
  numba._dispatcher.compute_fingerprint() returns the same bytes for both
  classes, so the dispatcher reuses the FIRST class's compiled overload for the
  second, and jit code that returns a bundle hands back an instance of the WRONG
  class. numba's own test suite states the invariant this breaks
  (numba/tests/test_typeof.py::TestFingerprint): "Each fingerprint must denote
  values of only one Numba type (this is the condition for correctness)".

So this file guards the trap with THREE tests, in descending order of power:

  1. test_bundle_registry_rejects_fingerprint_collision  -- exact, deterministic
  2. test_jit_returns_the_callers_bundle_class           -- exact, deterministic
  3. test_per_call_dispatch_stays_near_1us               -- doc 05 9.8 as written

Test 3 is kept because it still guards the OTHER dispatch regressions (bundle
width blow-up, an accidental typed.List/typed.Dict argument, a fall back to the
Python typing path), but it is explicitly documented as powerless against the
same-name trap. Tests 1 and 2 are what actually close criterion 8.

Run:  <repo>/.venv/bin/python -m pytest test_dispatch_regression.py -v
      <repo>/.venv/bin/python test_dispatch_regression.py          # no pytest
      DECIDER2_DISPATCH_BUDGET_US=6.0 ... -m pytest ...            # slower CI
"""

import collections
import os
import time
from functools import lru_cache
from statistics import median

import pytest
from numba import njit
from numba._dispatcher import compute_fingerprint
from pydantic import BaseModel

# Measured on numba 0.67.0 / py3.14.5 / llvmlite 0.49.0 (see stability.log):
#   typical 2-field-bundle dispatch          ~1.0-1.4 us
#   worst of 20 independent trials           2.446 us  <- a transient machine
#                                                         noise spike, not the
#                                                         collision: that trial's
#                                                         before/after ratio was
#                                                         the only one above 1.08
# 6.0 us is ~2.5x the worst observation actually seen on an otherwise-busy box,
# and still 2.5-4x below the 15-24 us regime doc 01 4c wants caught. Raise it on
# a shared CI runner rather than letting this test flake.
DISPATCH_BUDGET_US = float(os.environ.get("DECIDER2_DISPATCH_BUDGET_US", "6.0"))

N_CALLS = 2000
REPEATS = 9
WARMUP = 200


# --------------------------------------------------------------------------
# The guard decider2 should actually ship (doc 01 4c rules 1 and 2, plus the
# fingerprint check the measurement showed is the part with teeth).
# --------------------------------------------------------------------------

class BundleCollision(AssertionError):
    """Two distinct bundle classes are indistinguishable to numba's dispatcher."""


class BundleRegistry:
    """Registers every generated params bundle and refuses two distinct classes
    that numba's C fast path cannot tell apart.

    The check is on compute_fingerprint(), not on __name__, because the
    fingerprint is exactly the key the dispatcher caches on: it encodes the
    class __name__, the field names and the field types, and nothing else.
    """

    def __init__(self):
        self._by_fingerprint = {}

    def register(self, cls, sample):
        fp = compute_fingerprint(sample)
        prior = self._by_fingerprint.get(fp)
        if prior is not None and prior is not cls:
            raise BundleCollision(
                f"bundle {cls.__name__!r} is indistinguishable to numba from an "
                f"already-registered distinct class {prior.__name__!r}: both "
                f"fingerprint as {fp!r}. numba will silently reuse the first "
                f"class's compiled overload and box results as {prior.__name__}. "
                f"Give the bundles different field names or different class names."
            )
        self._by_fingerprint[fp] = cls
        return cls


def _sanitise(module_id):
    return "".join(ch if ch.isalnum() else "_" for ch in module_id)


@lru_cache(maxsize=None)
def bundle_for(model, module_id):
    """doc 01 4c: class name derived from the module id, memoised on the
    pydantic model object."""
    return collections.namedtuple(f"P_{_sanitise(module_id)}", list(model.model_fields))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def percall_us(fn, arg, n_calls=N_CALLS, repeats=REPEATS):
    for _ in range(WARMUP):
        fn(arg)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            fn(arg)
        samples.append((time.perf_counter() - t0) / n_calls * 1e6)
    return median(samples)


def fresh_kernel():
    @njit(cache=False)
    def k(t):
        return t.cutoff * t.weight
    return k


class ScoreParams(BaseModel):
    cutoff: float
    weight: float


def redefine_score_params():
    """What re-running a notebook cell does: a new class object, same name and
    fields."""
    class ScoreParams(BaseModel):
        cutoff: float
        weight: float
    return ScoreParams


# ==========================================================================
# 1. exact guard -- this is the one that closes doc 05 9.8
# ==========================================================================

def test_bundle_registry_accepts_distinct_module_ids():
    """doc 01 4c rule 1: names derived from unique module ids never collide."""
    reg = BundleRegistry()
    for module_id in ("risk.score", "fraud.flag", "limit.cap"):
        cls = bundle_for(ScoreParams, module_id)
        reg.register(cls, cls(1.0, 2.0))
    assert len(reg._by_fingerprint) == 3


def test_bundle_registry_rejects_fingerprint_collision():
    """POSITIVE CONTROL: the guard must actually fire on the trap."""
    reg = BundleRegistry()
    a = collections.namedtuple("P_risk_score", ["cutoff", "weight"])
    b = collections.namedtuple("P_risk_score", ["cutoff", "weight"])
    reg.register(a, a(1.0, 2.0))
    with pytest.raises(BundleCollision, match="indistinguishable"):
        reg.register(b, b(1.0, 2.0))


def test_registry_permits_the_documented_fix_of_different_field_names():
    """doc 01 4c: 'Different field names avoid it entirely.'"""
    reg = BundleRegistry()
    a = collections.namedtuple("P_risk_score", ["cutoff", "weight"])
    b = collections.namedtuple("P_risk_score", ["lo", "hi"])
    reg.register(a, a(1.0, 2.0))
    reg.register(b, b(1.0, 2.0))   # must not raise
    assert len(reg._by_fingerprint) == 2


def test_memoisation_is_idempotent_for_the_same_model_object():
    """doc 01 4c rule 2: the same module's bundle is not regenerated per build."""
    first = bundle_for(ScoreParams, "risk.score")
    for _ in range(50):
        assert bundle_for(ScoreParams, "risk.score") is first


# ==========================================================================
# 2. exact guard -- the consequence the trap actually has on numba 0.67
# ==========================================================================

def test_jit_returns_the_callers_bundle_class():
    """Under the same-name collision numba boxes the result as the FIRST class it
    saw, not the one the caller passed. Deterministic; no timing involved."""
    a = collections.namedtuple("P_boxcheck", ["cutoff", "weight"])
    b = collections.namedtuple("P_boxcheck", ["lo", "hi"])   # the fix applied

    @njit(cache=False)
    def identity(t):
        return t

    assert type(identity(a(1.0, 2.0))) is a
    assert type(identity(b(3.0, 4.0))) is b
    assert len(identity.signatures) == 2


def test_collision_causes_wrong_class_positive_control():
    """POSITIVE CONTROL for the test above: without the fix, numba really does
    return the wrong class. If this ever starts passing as 'correct', numba
    changed and test_jit_returns_the_callers_bundle_class needs revisiting."""
    a = collections.namedtuple("P_boxctl", ["cutoff", "weight"])
    b = collections.namedtuple("P_boxctl", ["cutoff", "weight"])
    assert compute_fingerprint(a(1.0, 2.0)) == compute_fingerprint(b(1.0, 2.0))

    @njit(cache=False)
    def identity(t):
        return t

    identity(a(1.0, 2.0))               # compile for `a` first
    got = identity(b(3.0, 4.0))
    assert tuple(got) == (3.0, 4.0)     # values survive
    assert type(got) is a               # ...but the class does not
    assert len(identity.signatures) == 1


# ==========================================================================
# 3. doc 05 9.8 as literally written -- kept, but documented as low-power
# ==========================================================================

def test_per_call_dispatch_stays_near_1us():
    """doc 05 9.8. NOTE: measured to have no power against the same-name trap on
    numba 0.67 (collided dispatch = 1.03x clean). It still guards bundle-width
    blow-up and accidental Python-typing fallbacks."""
    bundle = bundle_for(ScoreParams, "risk.score")
    k = fresh_kernel()
    us = percall_us(k, bundle(0.5, 2.0))
    assert us < DISPATCH_BUDGET_US, (
        f"per-call dispatch {us:.3f} us exceeds budget {DISPATCH_BUDGET_US} us"
    )


def test_notebook_rerun_does_not_degrade_dispatch():
    """doc 08 4.4 assumes re-running a cell makes dispatch bite. Measured: it does
    not -- 10 distinct same-named bundle classes live on one dispatcher still
    dispatch at ~1.1 us. The hazard is the wrong-class boxing above, not latency."""
    k = fresh_kernel()
    first_model = redefine_score_params()
    first_bundle = bundle_for(first_model, "nb.score")
    inst = first_bundle(0.5, 2.0)
    baseline = percall_us(k, inst)

    alive = [(first_model, first_bundle)]
    for _ in range(9):
        m = redefine_score_params()
        b = bundle_for(m, "nb.score")
        alive.append((m, b))
        k(b(1.0, 1.0))

    after = percall_us(k, inst)
    assert after < DISPATCH_BUDGET_US, (
        f"after 9 notebook re-runs dispatch is {after:.3f} us "
        f"(baseline {baseline:.3f} us), over budget {DISPATCH_BUDGET_US} us"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))
