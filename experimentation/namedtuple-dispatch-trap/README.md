# Experiment D — the same-name NamedTuple dispatch trap

## What this measures

Whether the hazard recorded in decider2 **doc 01 §4c** is real, and what shape it
actually has:

> Two **distinct** NamedTuple classes sharing the same `__name__` *and* the same
> field names blow per-call dispatch from ~1 µs to **15–24 µs — permanently, for
> every call involving that name.** The numba types print identically
> (`Z(float64 x 2)`) but compare unequal, so the dispatcher's cache thrashes.
> Different field names avoid it entirely.

and whether the regression test **doc 05 §9 criterion 8** asks for can actually
catch it:

> A regression test asserts per-call dispatch stays near 1 µs, guarding the
> same-name-NamedTuple trap (doc 01 §4c) that silently costs 15–24 µs forever.

## How to run

```sh
cd experimentation/namedtuple-dispatch-trap

# full measurement report (~60 s, mostly numba compilation)
../../.venv/bin/python experiment.py

# is the effect real or noise? N independent trials (~30 s)
../../.venv/bin/python stability.py 20

# the deliverable regression test (~1 s)
../../.venv/bin/python -m pytest test_dispatch_regression.py -v
../../.venv/bin/python test_dispatch_regression.py        # same, without pytest on PATH

# slower CI box
DECIDER2_DISPATCH_BUDGET_US=8.0 ../../.venv/bin/python -m pytest test_dispatch_regression.py
```

Nothing is installed, written or mutated. `run1.log`, `run2.log`, `run3.log` and
`stability.log` are captured outputs of independent runs.

## Method

`percall_us()` times a median of 9 repeats × 2000 calls after 200 warmup calls,
so compilation is fully out of the measurement. The measurement loop's own floor
(a no-op Python function) is measured at **0.048 µs/call** and reported, so the
~1 µs figures are 20× above the instrument's noise floor. Kernel bodies are
trivial (`t.x + t.y`) so dispatch dominates. Every scenario gets a fresh
dispatcher and globally unique class names so phases cannot contaminate each
other — except where a collision is the thing being measured.

## Stack

python 3.14.5, numba 0.67.0, llvmlite 0.49.0, numpy 2.4.6, pydantic 2.13.4.
**Every number below is from this stack only.** The doc's 15–24 µs was measured
on an unrecorded stack (E5); which numba version it used is not in the docs.

## Result: the performance claim is REFUTED; the trap is real but is a
## correctness defect, not a latency one

Per-call dispatch, 2-field float bundle (µs/call, median):

| scenario | µs/call | vs clean |
|---|---|---|
| clean, single class | 1.006 | 1.00× |
| **same `__name__` + same fields (the "trap")** | **1.039** | **1.03×** |
| same `__name__`, different fields (the doc's fix) | 1.085 | 1.08× |
| different `__name__`, same fields | 1.038 | 1.03× |
| prescribed fix, 50 pipeline builds | 1.175 | 1.17× |
| notebook, 9 cell re-runs, id-derived name | 1.084 | 1.08× |
| notebook, 9 cell re-runs, unique names | 1.386 | 1.38× |

decider2-shaped driver (5 bundles + 3 arrays): 5.478 µs clean → 5.700 µs
collided = **1.04×**.

**The 15–24 µs regime was not observed in any configuration.** The total spread
across every variant measured is 1.006–1.386 µs (1.38×), which is ordinary
per-argument dispatch cost, not a trap.

### Confirmed against run-to-run noise (`stability.py`, 40 independent trials)

One early run of `experiment.py` (`run3.log`) reported the collided case at
2.589 µs / 2.53×, which would have looked like partial support. It was noise.
`stability.py` runs the before/after reproduction as N independent trials, each
with a fresh dispatcher and a globally unique class name:

| batch of 20 trials | median before | median after | ratio (median of trials) | ratio range |
|---|---|---|---|---|
| batch 1 | 1.060 µs | 1.053 µs | **0.99×** | 0.84–2.32 |
| batch 2 | 1.070 µs | 1.041 µs | **0.99×** | 0.84–1.07 |

`signatures == 1` in **all 40 trials**. Batch 1's 2.32× is a single trial whose
*minimum* sample was 1.607 µs against a median of 2.446 µs — a transient CPU
contention spike (this box was running `uv add` during the session), not the
collision. 39 of 40 trials fall in 0.84–1.08×. The effect size is zero.

### The stated mechanism is half right

| doc 01 §4c says | measured |
|---|---|
| types print identically (`Z(float64 x 2)`) | **confirmed** — `str()` equal |
| types compare unequal | **confirmed** — `==` False, hashes differ, `instance_class` differs |
| "the dispatcher's cache **thrashes**" | **refuted** — `signatures` stays at **1**; no second overload is ever compiled, `cache_misses` shows one entry |

The reason there is no thrash is visible in `numba._dispatcher.compute_fingerprint`,
the key the C fast path caches on:

```
A -> b'Zc(xfyf)'
B -> b'Zc(xfyf)'      # identical
```

It encodes class `__name__` + field names + field types and nothing else, so the
fast path cannot tell the two classes apart — and therefore never misses. It
serves the **first** class's compiled overload to the second, at full speed.

### What it costs instead

numba's own test suite states the invariant this breaks
(`numba/tests/test_typeof.py::TestFingerprint`):

> "Each fingerprint must denote values of only one Numba type (this is the
> condition for correctness)"

Measured consequence — compile a jit function for class `A`, then call it with
its same-named twin `B`:

```
identity(RB(3.0, 4.0)) -> Wbox(x=3.0, y=4.0)
  values correct?    True
  type(got) is RB ?  False     <- the class the caller passed
  type(got) is RA ?  True      <- the FIRST class numba saw
  signatures after = 1
```

**Any jit function returning a params bundle hands back an instance of the wrong
class, silently.** Values are unaffected (namedtuples are positional tuples), so
this only bites code that does `isinstance` / `type(...) is` / attribute-name
dispatch on a returned bundle. With different field names (the doc's fix) the
same test returns the caller's class and compiles 2 signatures — **the fix works,
for the correctness problem rather than the performance one.**

### Permanence and scope (also refuted, because there is nothing to be permanent)

| probe | µs/call |
|---|---|
| immediately after the collision | 1.039 |
| after 20,000 further calls with A only | ~1.0 |
| after `del B`, `gc.collect()` | ~1.0 |
| after `dispatcher._reset_overloads()` | ~1.0 |
| a **fresh** dispatcher, called with the tainted class | 1.058 |
| a fresh dispatcher, clean class | 1.065 |

No degradation to recover from, and nothing global: a dispatcher that never saw
the twin behaves identically.

### The notebook wrinkle (doc 08 §4.4) does not bite the way it assumes

Redefining the pydantic model 1/2/3/5/9 times, with the bundle name derived from
an unchanged module id, produces 10 distinct classes all named `P_nb_score` on one
dispatcher. Dispatch on the original bundle: **1.161 → 1.084 µs**, `signatures`
stays 1 throughout. Flat.

Interestingly the *mitigation* is the slower option: giving each generation a
unique name (`P_nbu_score_1..10`) forces 10 real signatures and dispatch rises
1.193 → **1.386 µs** — a genuine linear cost in overload count. It buys back the
type-identity correctness, and it costs 10 compilations, but it is not free.

## Deliverable: the CI threshold

Doc 05 §9.8 as written **cannot** close its own criterion on this stack: the
thing it wants caught costs 1.03×, so no timing budget separates pass from fail.
`test_dispatch_regression.py` therefore guards it three ways:

1. `BundleRegistry` — rejects two distinct bundle classes whose
   `compute_fingerprint` collides. Exact, deterministic, zero flake, and it is
   the *same key the dispatcher uses*, so it cannot drift from the real hazard.
   Ships with a positive control that proves it fires.
2. A boxing-identity test — jit must return the caller's class. Deterministic,
   with a positive control asserting the un-fixed case really does return the
   wrong class.
3. The literal doc 05 §9.8 timing assertion, kept for the regressions it *can*
   see (bundle width blow-up, an accidental `typed.List`/`typed.Dict` argument),
   and documented in-file as powerless against the same-name trap.

**Measured threshold for (3): `DISPATCH_BUDGET_US = 6.0`.** Typical 2-field
dispatch is 1.0–1.4 µs, but the worst single observation across 40 trials on a
busy box was **2.446 µs** — from machine noise, not from any code path. 6.0 µs is
~2.5× that worst observation and still 2.5–4× below the 15–24 µs regime doc 01
§4c wants caught. Overridable via `DECIDER2_DISPATCH_BUDGET_US`; raise it on a
shared runner rather than letting the test flake.

This budget is for a **2-field single-bundle** probe. A 20-field bundle measures
2.864 µs and the 8-argument driver 5.478 µs, so any budget must be pinned to a
fixed probe shape, not applied to arbitrary kernels.

## Incidental measurements

| argument shape | µs/call |
|---|---|
| scalar float | 0.248 |
| 1-D float64 array | 0.314 |
| 2-field namedtuple | 1.022 |
| 20-field namedtuple | 2.864 |
| `typed.List` | 1.786 |
| `typed.Dict` | 1.814 |

`compute_fingerprint` **fails** (`NotImplementedError`) for `dict`, `typed.List`
and `typed.Dict`, forcing the Python typing path — but that path costs only
~1.8 µs here, so it is not the origin of the doc's 15–24 µs either.

## What was not tested

- Older numba versions. The refutation is scoped to numba 0.67.0; it is
  plausible the doc's 15–24 µs was real on whatever version E5 ran. Confirming
  that would need a second environment, which this task forbids installing.
- Bundle field types beyond float64 (int, bool, nested tuple, array fields) in
  the collision case — only the fingerprint strings were inspected for those.
- `cache=True` dispatchers and on-disk cache interaction.
