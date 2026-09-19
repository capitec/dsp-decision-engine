# Experiment M — the K/L contradiction, resolved

Settles EXPERIMENTS.md's "The K/L contradiction — unresolved". §K
(`subprocess-handover/`) and §L (`rule-thresholds-as-args/`) ran the same
stale-constant A/B and reached opposite attributions: §K said clearing
CPython's `__pycache__/*.pyc` alone fixed it (CPython's fault); §L said with
the `.pyc` cleared and numba's own `.nbi`/`.nbc` untouched, it still served
stale (numba's fault).

**Both are right, about different edit shapes. It is not a harness bug in
either one.**

## What it measures

1. **`run_2x2.py`** — a clean 2×2 (both caches live / only `.pyc` cleared /
   only numba's `.nbi`+`.nbc` cleared / both cleared), using **real
   subprocesses** (matching §K's methodology and doc 08 §4's actual
   deployment shape — compile in one process, load in another) and
   **module-name import** (`sys.path` + `import drv`, doc 05 §4.1's own
   fix — never `spec_from_file_location`). The driver (`gen.py`) has the
   threshold literal appear **exactly once** in the whole function — §L's
   shape, and the shape an actual per-rule threshold UI edit produces.
2. **`verify_mechanism.py`** — direct `co_consts`/`co_code` comparison
   (no numba, pure `compile()`) for three cases: §L's shape (threshold
   unique), a template-regenerated §K-shape (all 16 occurrences of the
   shared constant changed at once), and **§K's actual edit** (only the
   `a1`-branch occurrence of the shared `0.5` changed, the other 15 left
   alone — literally reproducing what
   `numba_cache_survival/run_experiment.py:part_stale` and §K's harness do).
   Also dumps the `.nbi` index's condition-6 hash tuple directly.
3. **`content_addressed.py`** — the proposed fix, both caches left
   completely untouched (worst case), filename **and** module name derived
   from `sha256(source)[:16]`.

## Results (measured; `results.jsonl` has the raw rows)

### The 2×2 (real subprocess, real module import, §L's driver shape)

| arm | `.pyc` cleared | numba cache cleared | value served | ground truth | stale? |
|---|---|---|---|---|---|
| both live | no | no | 0.1 | 0.15 | **yes** |
| `.pyc` only cleared | **yes** | no | **0.1** | 0.15 | **yes** |
| numba cache only cleared | no | yes | 0.1 | 0.15 | **yes** |
| both cleared | yes | yes | 0.15 | 0.15 | no |

The load-bearing row is **"`.pyc` only cleared"**: `child.py`'s own dump of
`drv.driver.py_func.__code__.co_consts` after that run shows
`(0.595, 2.0, 3.0)` — CPython genuinely re-parsed the *edited* source, not
a cached bytecode object. The value served was still **0.1, the pre-edit
answer**. The only thing left that could have produced that is numba's own
on-disk cache (`.nbi`/`.nbc`, untouched in this arm), matched by
`(mtime, size)` and a `co_code` hash that hadn't changed. This reproduces
§L's finding **under §K's own methodology** (real subprocess boundary, not
§L's in-process `spec_from_file_location` reload) — so the disagreement was
never about which harness's cache-clearing code was buggy; both cleared
exactly what they said they cleared.

### The mechanism (`verify_mechanism.py`)

| case | co_consts before | co_consts after | co_code identical? |
|---|---|---|---|
| §L shape (threshold unique) | `(0.0405, 2.0, 3.0)` | `(0.595, 2.0, 3.0)` | **True** |
| §K shape, template-regenerated (all 16 occurrences changed) | 9 entries | 9 entries (same length) | True |
| **§K's actual edit** (1 of 16 occurrences changed, `0.5` still needed elsewhere) | 9 entries | **10 entries** | **False** |

§K's real edit *inserts* a new `co_consts` slot for `0.7` while *keeping*
the old slot for `0.5` (still referenced by the other 7 branches) — every
`LOAD_CONST` operand after the insertion point shifts, so `co_code` changes
and numba correctly misses. §K's driver shares one literal across many
branches, so its "same-byte-length" edit (changing one occurrence, per
`numba_cache_survival/run_experiment.py:part_stale`'s own trick) is
*structurally* incapable of leaving `co_code` unchanged — §K could not
construct the isolating case not because it doesn't exist, but because its
driver's shape forecloses it. §L's per-rule-threshold shape (and the real
business-UI scenario doc 05 §4.2 is about) has no such collision, and
`co_code` stays byte-identical, which is exactly condition 6's blind spot.

The `.nbi` index dump confirms condition 6 verbatim: the stored key hashes
`co_code` and the pickled closure only — `('6761545c...', 'e3b0c442...')`
(the second hash is `sha256(b"")`, i.e. no closure) — `co_consts` is not in
the key at all, at either layer's cache.

### Content-addressed naming (`content_addressed.py`), neither cache cleared

| step | file | value served | correct? |
|---|---|---|---|
| baseline | `drv_bc366282cf96f7af.py` | 0.1 | yes |
| unchanged "redeploy" (same content) | same path (write skipped) | 0.1 | yes — honest hit |
| edited content | `drv_<new hash>.py` (different path) | 0.15 | yes |

No cache was cleared in any of these three calls. A different hash means a
different path (and a different module-registration name), so there is no
`(mtime, size)` pair left for either layer to coincide on — this closes the
hazard at **both** layers at once, confirmed against the layer this
experiment shows is actually responsible (numba's own cache, independently
of CPython's).

## Verdict

**Both §K and §L are correct, about different edit shapes — this is not
"one harness had a bug."** CPython's `__pycache__/*.pyc` is *sufficient on
its own* to cause the hazard (it can feed numba an unchanged function
object without numba ever seeing the edit). Numba's own on-disk cache is
*independently sufficient* on its own, for the specific — and realistic —
case where the edited constant has no other occurrence in the function,
which is exactly what a per-rule threshold looks like. §K's driver
happened to share every literal across many branches, which meant every
edit it could try also happened to change `co_code` through slot
insertion, masking numba's own exposure. Content-addressed naming is
confirmed to close both, without relying on either cache being cleared
correctly by a build tool.

## Run

```
/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/.venv/bin/python run_2x2.py
/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/.venv/bin/python verify_mechanism.py
/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/.venv/bin/python content_addressed.py
```

Total wall clock, all three: **under 15 s**. Peak memory: negligible (a
handful of KB — one scalar njit function, no arrays); run in the
foreground, `free -g` checked before starting (17-22 GB available
throughout), well clear of the 8 GB floor and nowhere near the 2 GB
tmux/systemd-run threshold.

## Dropped, for the time budget

- Re-testing §K's "seventh condition" (`sys.modules` registration name) as
  an axis of this same 2×2 — not needed to resolve the K/L disagreement
  itself, which is about conditions 3 and 6, not 7.
- A nested/multi-rule version of the §L driver — the single-literal driver
  is sufficient to isolate the mechanism; scaling it would only restate §L's
  own rule-count results.
