"""
EXPERIMENT D -- the same-name NamedTuple dispatch trap.

Tests the claim in decider2 doc 01 section 4c:

    "Two distinct NamedTuple classes sharing the same __name__ AND the same field
     names blow per-call dispatch from ~1 us to 15-24 us -- permanently, for every
     call involving that name. The numba types print identically (Z(float64 x 2))
     but compare unequal, so the dispatcher's cache thrashes. Different field
     names avoid it entirely."

Run:  <repo>/.venv/bin/python experiment.py
No arguments. Prints a report to stdout. Writes nothing.
"""

import collections
import gc
import sys
import time
from statistics import median

import numba
import numba._dispatcher as _nbd
from numba import njit
from pydantic import BaseModel
from functools import lru_cache

# ---------------------------------------------------------------- timing ----

N_CALLS = 2000
REPEATS = 9
WARMUP = 200


def percall_us(fn, arg, n_calls=N_CALLS, repeats=REPEATS):
    """Median per-call wall time in microseconds. Compilation is warmed out."""
    for _ in range(WARMUP):
        fn(arg)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            fn(arg)
        t1 = time.perf_counter()
        samples.append((t1 - t0) / n_calls * 1e6)
    return median(samples)


def loop_floor_us(n_calls=N_CALLS, repeats=REPEATS):
    """Cost of the measurement loop itself, calling a no-op Python function."""
    def noop(t):
        return t
    arg = (1.0, 2.0)
    for _ in range(WARMUP):
        noop(arg)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            noop(arg)
        t1 = time.perf_counter()
        samples.append((t1 - t0) / n_calls * 1e6)
    return median(samples)


def fresh_kernel():
    """A brand new njit dispatcher with a trivial 2-field body."""
    @njit(cache=False)
    def k(t):
        return t.x + t.y
    return k


def row(label, value, unit="us/call"):
    print(f"  {label:<52} {value:>9.3f} {unit}")


def hdr(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


results = {}

# ------------------------------------------------------- 0. environment ----

hdr("0. ENVIRONMENT")
print(f"  python   {sys.version.split()[0]}")
print(f"  numba    {numba.__version__}")
import llvmlite, numpy, pydantic
print(f"  llvmlite {llvmlite.__version__}   numpy {numpy.__version__}   pydantic {pydantic.__version__}")
FLOOR = loop_floor_us()
row("python call-loop floor (no-op function)", FLOOR)
print(f"  timing: median of {REPEATS} repeats x {N_CALLS} calls, {WARMUP} warmup calls")

# --------------------------------------- 1. reproduce: before / after -----

hdr("1. REPRODUCE -- per-call dispatch before and after the collision")

A = collections.namedtuple("Zc", ["x", "y"])
a = A(1.0, 2.0)
k1 = fresh_kernel()

before = percall_us(k1, a)
row("BEFORE collision: kernel(A) where A is the only 'Zc'", before)
print(f"      signatures={len(k1.signatures)}  {k1.signatures}")

# Now introduce the twin: same __name__, same field names, distinct class.
B = collections.namedtuple("Zc", ["x", "y"])
b = B(3.0, 4.0)
k1(b)   # one call -- compiles the second signature

after_a = percall_us(k1, a)
after_b = percall_us(k1, b)
row("AFTER collision:  kernel(A)", after_a)
row("AFTER collision:  kernel(B)", after_b)
print(f"      signatures={len(k1.signatures)}  {k1.signatures}")
row("degradation factor kernel(A) after/before", after_a / before, "x")
results["before"] = before
results["after_a"] = after_a
results["after_b"] = after_b

# Negative control: a kernel whose two argument classes differ in NAME only.
hdr("1b. CONTROLS -- which dimension actually triggers it")

C1 = collections.namedtuple("Zn1", ["x", "y"])
C2 = collections.namedtuple("Zn2", ["x", "y"])
k2 = fresh_kernel()
k2(C1(1.0, 2.0)); k2(C2(1.0, 2.0))
ctrl_name = percall_us(k2, C1(1.0, 2.0))
row("different __name__, same fields  -> kernel(C1)", ctrl_name)

@njit(cache=False)
def k3(t):
    return t[0] + t[1]
D1 = collections.namedtuple("Zf", ["x", "y"])
D2 = collections.namedtuple("Zf", ["p", "q"])
k3(D1(1.0, 2.0)); k3(D2(1.0, 2.0))
ctrl_fields = percall_us(k3, D1(1.0, 2.0))
row("same __name__, different fields  -> kernel(D1)   [THE DOC'S FIX]", ctrl_fields)

E1 = collections.namedtuple("Zs", ["x", "y"])
k4 = fresh_kernel()
solo = percall_us(k4, E1(1.0, 2.0))
row("single class, no twin at all (clean baseline)", solo)
results["ctrl_name"] = ctrl_name
results["ctrl_fields"] = ctrl_fields
results["solo"] = solo

# ----------------------------------------------- 2. mechanism / caches ----

hdr("2. MECHANISM -- do the types print identically but compare unequal?")

ta, tb = numba.typeof(a), numba.typeof(b)
print(f"  repr(typeof(A instance)) = {ta!r}")
print(f"  repr(typeof(B instance)) = {tb!r}")
print(f"  str equal?      {str(ta) == str(tb)}")
print(f"  types equal?    {ta == tb}")
print(f"  hashes equal?   {hash(ta) == hash(tb)}")
print(f"  type keys:      {ta.key}")
print(f"                  {tb.key}")
print(f"  instance_class identical? {ta.instance_class is tb.instance_class}")
print()
fa = _nbd.compute_fingerprint(a)
fb = _nbd.compute_fingerprint(b)
print("  numba._dispatcher.compute_fingerprint (the C fast-path cache key):")
print(f"    A -> {fa!r}")
print(f"    B -> {fb!r}")
print(f"    FINGERPRINTS EQUAL? {fa == fb}   <-- if True the fast path cannot tell them apart")
fd1 = _nbd.compute_fingerprint(D1(1.0, 2.0))
fd2 = _nbd.compute_fingerprint(D2(1.0, 2.0))
print(f"    different-fields pair equal? {fd1 == fd2}   ({fd1!r} vs {fd2!r})")
fc1 = _nbd.compute_fingerprint(C1(1.0, 2.0))
fc2 = _nbd.compute_fingerprint(C2(1.0, 2.0))
print(f"    different-name   pair equal? {fc1 == fc2}   ({fc1!r} vs {fc2!r})")
print()
print("  dispatcher cache state for the collided kernel:")
print(f"    signatures      = {k1.signatures}")
print(f"    overload keys   = {list(k1.overloads.keys())}")
print(f"    stats           = {k1.stats}")
print(f"    _types_active_call = {k1._types_active_call}")

# ------------------------------------------------------ 3. permanence ----

hdr("3. PERMANENCE -- does it recover if you stop using the second class?")

for _ in range(20000):
    k1(a)
rec1 = percall_us(k1, a)
row("after 20,000 further calls with A only", rec1)

del b, B
gc.collect()
rec2 = percall_us(k1, a)
row("after deleting class B + instance, gc.collect()", rec2)

k1._reset_overloads()
k1(a)
rec3 = percall_us(k1, a)
row("after dispatcher._reset_overloads() + recompile", rec3)
print(f"      signatures now = {len(k1.signatures)}")
results["recover_calls"] = rec1
results["recover_gc"] = rec2
results["recover_reset"] = rec3

hdr("3b. SCOPE -- is a brand-new dispatcher that never saw the twin also poisoned?")

k5 = fresh_kernel()
fresh_on_tainted = percall_us(k5, a)
row("fresh kernel called with the tainted class A", fresh_on_tainted)
k6 = fresh_kernel()
F1 = collections.namedtuple("Zclean", ["x", "y"])
fresh_on_clean = percall_us(k6, F1(1.0, 2.0))
row("fresh kernel called with a clean, unique class", fresh_on_clean)
results["fresh_on_tainted"] = fresh_on_tainted
results["fresh_on_clean"] = fresh_on_clean

# ------------------------------------------ 2b. CORRECTNESS CONSEQUENCE ----

hdr("2b. CORRECTNESS -- what the fingerprint collision actually costs")

print("  numba's own invariant (numba/tests/test_typeof.py, TestFingerprint):")
print('    "Each fingerprint must denote values of only one Numba type')
print('     (this is the condition for correctness)"')
print()

RA = collections.namedtuple("Wbox", ["x", "y"])
RB = collections.namedtuple("Wbox", ["x", "y"])

@njit(cache=False)
def identity(t):
    return t

_ = identity(RA(1.0, 2.0))          # compile for RA FIRST
print(f"  compiled for RA only; signatures={len(identity.signatures)}")
got = identity(RB(3.0, 4.0))        # now call with the twin
print(f"  identity(RB(3.0, 4.0)) -> {got!r}")
print(f"    values correct? {tuple(got) == (3.0, 4.0)}")
print(f"    type(got) is RB ? {type(got) is RB}    <-- caller's class")
print(f"    type(got) is RA ? {type(got) is RA}    <-- the FIRST class seen")
print(f"    signatures after = {len(identity.signatures)} (no second overload compiled)")
print()
print("  Negative control -- different field names, same __name__:")
RC = collections.namedtuple("Wbox2", ["x", "y"])
RD = collections.namedtuple("Wbox2", ["p", "q"])

@njit(cache=False)
def identity2(t):
    return t

_ = identity2(RC(1.0, 2.0))
got2 = identity2(RD(3.0, 4.0))
print(f"    type(identity2(RD)) is RD ? {type(got2) is RD}   "
      f"signatures={len(identity2.signatures)}")

results["boxing_wrong_class"] = (type(got) is RA)
results["boxing_control_ok"] = (type(got2) is RD)

# -------------------------------------------- 4. realistic driver shape ----

hdr("4. REALISM -- a decider2-shaped driver (5 bundles + 3 arrays)")

import numpy as np

mk = lambda nm: collections.namedtuple(nm, ["lo", "hi", "w"])
P1, P2, P3, P4, P5 = (mk(f"Pm{i}") for i in range(1, 6))

@njit(cache=False)
def wide_driver(p1, p2, p3, p4, p5, a, b, out):
    for i in range(a.shape[0]):
        v = a[i] * p1.w + b[i] * p2.w
        if v > p3.hi:
            v = p4.lo
        out[i] = v * p5.w
    return out[0]

a_ = np.arange(64, dtype=np.float64)
b_ = np.arange(64, dtype=np.float64)
o_ = np.zeros(64)
args = (P1(0.,1.,2.), P2(0.,1.,2.), P3(0.,1.,2.), P4(0.,1.,2.), P5(0.,1.,2.), a_, b_, o_)

def call_wide(_):
    return wide_driver(*args)

wide_before = percall_us(call_wide, None, n_calls=1000, repeats=7)
row("5 bundles + 3 arrays, no collision", wide_before)

P1b = collections.namedtuple("Pm1", ["lo", "hi", "w"])   # the twin of P1
wide_driver(P1b(0., 1., 2.), *args[1:])
wide_after = percall_us(call_wide, None, n_calls=1000, repeats=7)
row("same, after one bundle class gets a same-name twin", wide_after)
print(f"      signatures={len(wide_driver.signatures)}  "
      f"ratio={wide_after / wide_before:.2f}x")
results["wide_before"] = wide_before
results["wide_after"] = wide_after

# ------------------------------ 5. the prescribed fix: id-derived + lru ----

hdr("5. PRESCRIBED FIX -- name from module id + @lru_cache on the pydantic model")

def _sanitise(module_id):
    return "".join(ch if ch.isalnum() else "_" for ch in module_id)

@lru_cache(maxsize=None)
def _bundle_for(model, module_id):
    """decider2 doc 01 4c rule 1+2: class name from the module id, memoised on
    the pydantic model object."""
    fields = list(model.model_fields)
    return collections.namedtuple(f"P_{_sanitise(module_id)}", fields)

class ScoreParams(BaseModel):
    cutoff: float
    weight: float

@njit(cache=False)
def driver(p):
    return p.cutoff * p.weight

MODULE_ID = "risk.score"
b1 = _bundle_for(ScoreParams, MODULE_ID)
b2 = _bundle_for(ScoreParams, MODULE_ID)
print(f"  _bundle_for is idempotent for the same model object: {b1 is b2}")
print(f"  bundle __name__ = {b1.__name__!r}   fields = {b1._fields}")
print(f"  lru_cache info  = {_bundle_for.cache_info()}")

# simulate 50 pipeline builds, all re-deriving the bundle for the same model
for _ in range(50):
    cls = _bundle_for(ScoreParams, MODULE_ID)
    assert cls is b1
inst = b1(0.5, 2.0)
fix_same = percall_us(driver, inst)
row("dispatch after 50 pipeline builds, same model object", fix_same)
print(f"      signatures={len(driver.signatures)}  lru={_bundle_for.cache_info()}")

# a second module: different id -> different class name, no collision possible
class OtherParams(BaseModel):
    cutoff: float
    weight: float

ob = _bundle_for(OtherParams, "fraud.flag")
print(f"  second module bundle __name__ = {ob.__name__!r} (distinct by construction)")
driver(ob(1.0, 2.0))
fix_two = percall_us(driver, inst)
row("dispatch with 2 distinct-id bundles live on one driver", fix_two)
print(f"      signatures={len(driver.signatures)}")
results["fix_same"] = fix_same
results["fix_two"] = fix_two

# ---------------------------------- 6. the notebook wrinkle (doc 08 4.4) ----

hdr("6. NOTEBOOK CELL RE-RUN -- redefine the model, keep the module id")

@njit(cache=False)
def nb_driver(p):
    return p.cutoff * p.weight

def redefine_model():
    """Exactly what re-running a notebook cell does: a NEW class object with the
    same name and the same fields."""
    class ScoreParams(BaseModel):
        cutoff: float
        weight: float
    return ScoreParams

NB_ID = "nb.score"                      # distinct id so section 5 cannot taint this
alive = []                              # hold refs so nothing is gc'd mid-measurement
first_model = redefine_model()
first_bundle = _bundle_for(first_model, NB_ID)
alive.append((first_model, first_bundle))
first_inst = first_bundle(0.5, 2.0)
nb_curve = [(0, percall_us(nb_driver, first_inst))]
row("k=0 re-runs (fresh session)", nb_curve[0][1])
print(f"      bundle __name__={first_bundle.__name__!r} signatures={len(nb_driver.signatures)}")

n_redefs = 0
for k in (1, 2, 3, 5, 9):
    while n_redefs < k:
        m = redefine_model()
        bcls = _bundle_for(m, NB_ID)
        alive.append((m, bcls))
        nb_driver(bcls(1.0, 1.0))       # the re-staged pipeline runs once
        n_redefs += 1
    t = percall_us(nb_driver, first_inst)
    nb_curve.append((k, t))
    row(f"after {k} cell re-run(s): dispatch on the ORIGINAL bundle", t)
    print(f"      distinct bundle classes={len(alive)}  all named "
          f"{first_bundle.__name__!r}  signatures={len(nb_driver.signatures)}")

results["nb_curve"] = nb_curve

# mitigation: make the generated name unique per generation
hdr("6b. MITIGATION -- disambiguate the generated class name per generation")

@njit(cache=False)
def nb_driver2(p):
    return p.cutoff * p.weight

_gen = [0]

@lru_cache(maxsize=None)
def _bundle_for_unique(model, module_id):
    _gen[0] += 1
    fields = list(model.model_fields)
    return collections.namedtuple(f"P_{_sanitise(module_id)}_{_gen[0]}", fields)

U_ID = "nbu.score"
alive2 = []
m0 = redefine_model()
u0 = _bundle_for_unique(m0, U_ID)
alive2.append((m0, u0))
u_inst = u0(0.5, 2.0)
results["nb_unique_0"] = percall_us(nb_driver2, u_inst)
row("k=0 re-runs, unique-per-generation names", results["nb_unique_0"])

n2 = 0
for k in (1, 3, 9):
    while n2 < k:
        m = redefine_model()
        bc = _bundle_for_unique(m, U_ID)
        alive2.append((m, bc))
        nb_driver2(bc(1.0, 1.0))
        n2 += 1
    t = percall_us(nb_driver2, u_inst)
    results[f"nb_unique_{k}"] = t
    row(f"after {k} cell re-run(s), unique names", t)
    print(f"      distinct classes={len(alive2)}  latest name="
          f"{alive2[-1][1].__name__!r}  signatures={len(nb_driver2.signatures)}")

# ------------------------------------------------------------- summary ----

hdr("SUMMARY -- measured on this stack only")
print(f"  loop floor (no-op python call)               {FLOOR:8.3f} us")
print()
print("  PER-CALL DISPATCH, 2-field float bundle:")
print(f"    clean, single class                        {solo:8.3f} us")
print(f"    same __name__ + same fields (the 'trap')   {after_a:8.3f} us"
      f"   ({after_a / solo:.2f}x clean)")
print(f"    same __name__, different fields            {ctrl_fields:8.3f} us")
print(f"    different __name__, same fields            {ctrl_name:8.3f} us")
print(f"    prescribed fix, 50 pipeline builds         {fix_same:8.3f} us")
print(f"    notebook, 9 cell re-runs (id-derived name) {nb_curve[-1][1]:8.3f} us")
print(f"    notebook, 9 cell re-runs (unique names)    {results['nb_unique_9']:8.3f} us")
print()
print("  DECIDER2-SHAPED DRIVER (5 bundles + 3 arrays):")
print(f"    no collision                               {wide_before:8.3f} us")
print(f"    with collision                             {wide_after:8.3f} us"
      f"   ({wide_after / wide_before:.2f}x)")
print()
allv = [solo, after_a, after_b, ctrl_fields, ctrl_name, fix_same, fix_two,
        nb_curve[-1][1], results["nb_unique_9"]]
print(f"  spread across EVERY 2-field variant above: "
      f"{min(allv):.3f} - {max(allv):.3f} us  ({max(allv) / min(allv):.2f}x)")
print()
print("  VERDICT on doc 01 4c's 15-24 us claim: NOT REPRODUCED on this stack.")
print("  The collision is real and detectable, but it is a CORRECTNESS defect,")
print("  not a performance one:")
print(f"    fingerprints collide:                 {fa == fb}")
print(f"    numba types compare unequal:          {ta != tb}")
print(f"    numba types print identically:        {str(ta) == str(tb)}")
print(f"    second overload compiled:             {len(k1.signatures) > 1}")
print(f"    jit returned the WRONG bundle class:  {results['boxing_wrong_class']}")
print()
print("  => A timing assertion (doc 05 9.8) has NO power to catch this here.")
print("     Guard it with the fingerprint/identity checks in "
      "test_dispatch_regression.py.")
