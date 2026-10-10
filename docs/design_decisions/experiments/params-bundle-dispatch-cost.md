# What does passing params bundles to a kernel cost, and are same-named bundle classes a trap?

Date: 2026-09-23 · Script: spikes, in decider-experimentation/ `e11-param-in-signature`, `namedtuple-dispatch-trap`

## Purpose
A kernel receives each step's params as a namedtuple bundle. Does the number of bundles or a name collision between two bundle classes slow every call, and which rule keeps the compiled signature stable?

## Method
Per-call dispatch of a 1-row driver (median of 2,000 calls after warmup, so typing the arguments dominates), numba 0.67. Each collision variant ran in its own subprocess so one could not poison the next. 20 steps each with a param `cap` and its own generated bundle; 40 independent trials of a before/after collision with a fresh dispatcher and a unique class name; fingerprints read from `numba._dispatcher.compute_fingerprint`.

## Result
| 20 bundles in one call | µs per call |
|---|---|
| 20 distinct classes, all named `Params`, field `cap` | 281.8 |
| class named after the step, or distinct field names | 18.9-19.6 |
| one flat bundle of 20 fields | 3.50 |
| 20 bundles nested in one outer tuple | 18.8 |
| 20 plain float arguments | 1.64 |

- The 15-24 µs slowdown the older design notes attributed to two same-named classes with the same fields did not reproduce when one class is passed per call: clean 1.006 µs, same name and fields 1.039 (1.03x), 40 trials with a median ratio of 0.99x (one 2.32x outlier was a busy box). The slow row above needs several distinct same-named classes in one call's arguments, and it does not persist: a clean driver measured 1.40 µs before and 1.43 after.
- It is a correctness defect, not a latency one. The C fast path fingerprints a bundle as class `__name__` + field names + field types (`Zc(xfyf)` for both classes), so it hands the second class the first one's overload and a jitted function that returns a bundle returns an instance of the wrong class. Different field names, or different class names, compile a second signature and return the right class.
- Each bundle costs about 0.95 µs to type per call, however it is named or nested; a scalar costs 0.08. Merging 20 bundles into one flat bundle is 5.4x cheaper (3.50 vs 18.9 µs; 5.74 µs at 40 fields). A 20-field bundle alone is 2.9 µs, a 2-field one 1.0.
- Retuning 15 values across the declared range kept one signature; redeclaring the param as `int` compiled a second. A sentinel default on the step's `param()` signature is harmless to numba; the dispatcher raises only if it is called without the param.
- A timing budget cannot guard the collision (it costs 1.03x, so no threshold separates pass from fail); the guard has to compare the fingerprint of every bundle class.

## Decision
One bundle class per field tuple (`bundle_class` caches it and names it from the fields), so two live classes never share a name and field names; the guard is that construction, not a timing assertion. Merging bundles per call is a measured option (3.50 vs 18.9 µs at 20 steps) that has not been needed: a `score()` is 28-35 µs with the params lookup at 1.5 µs. See [engine](../decisions/engine.md), `decider/_engine/params/bundles.py`.
