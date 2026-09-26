# Why a `Raw[str]` code cost 34 µs on one record, and what it costs now

`notes/strings.md` measured a one-step `Raw[str]` pipeline at 56.3 µs p50 on a single
record against a 22.2 µs numeric-kernel ceiling. The whole point of an int32 dictionary
code is that it sits in the fused array kernel, so paying that much to compare one
integer was backwards. None of it was the kernel.

## Where the time went

Measured by stubbing one stage at a time on the same box (`p50` of 3000 `score()`
calls, one-step `sector == PRIVATE` pipeline, `mode="fused"`):

| stage | µs |
|---|---|
| `StringCodes.encode` | 8.6 |
| per-read annotation introspection in `_run` and `_typed` | 5.5 |
| `State.representation` bookkeeping plus the closure built per call | 3.8 |
| the object-dtype record block and reading an object array | 1.7 |
| **total over the numeric ceiling** | **19.6** |

The individual calls, timed on their own:

| call | µs |
|---|---|
| `StringCodes.encode`, one value | 3.57 |
| `representation_for(Raw[str])` | 2.12 |
| `base_annotation(Raw[str])` | 1.11 |
| `np.fromiter`, one value | 0.54 |
| `is_raw(Raw[str])` | 0.26 |
| `raw_string_codes()` (a copy of the global table) | 0.14 |

Two findings behind those numbers:

- **`encode` did four passes to convert one value.** It took a lock, copied the
  process-global code table and merged it in, built a dict of every value by
  `setdefault`, then walked the values again through a generator into `np.fromiter`.
- **Every read re-derived what it already knew.** `rows_item`, `base_annotation`,
  `is_raw`, `representation_for` and an `any(c.node.kind == "row" ...)` scan ran per
  read per call, on `typing` introspection that cannot change once a plan is compiled.
  A `typing.get_origin` call is ~0.25 µs and the single-record path made ten of them.

## What changed

- One code table per process. `raw_str()` and the runtime encoder now draw from the
  same dict in `decider/types.py`, so `encode` has nothing to merge and nothing to
  copy, and `StringCodes` is gone. The fast path is one `dict.get`; the lock is taken
  only to assign a code to a string never seen before.
- `encode` became `representations.codes`: `np.fromiter(map(string_code, values), ...)`,
  one pass.
- `_external` decides once per plan how each read becomes what the kernel takes, and
  hands `_run` a `(kind, build)` pair — the representation key and a builder closure
  made at compile time. `_run` does no annotation introspection and builds no closure.
  `_typed` and `build_raw` are gone with it.
- `spans()`'s few-rows path, from `notes/strings-spans.md`, cherry-picked onto this
  structure: it borrows each value's own UTF-8 bytes through `PyBytes_AsString` instead
  of laying out a fresh buffer with `cumsum`/`stack`. Its `lru_cache` on the annotation
  helpers stays, for the compile and wiring paths that still call them per plan; the run
  path no longer calls them at all. The `"is a string input"` `TypeError` that `_typed`
  used to wrap moved to `_spans`, where `_external` now reaches `spans()`.

## Before and after

Same box, same script, interleaved rounds, best p50 of four (`mode="fused"`,
200k rows for the batch column). "before" is `d825f31`: neither this change nor the
spans one.

| variant | p50 before | p50 after | rows/s before | rows/s after |
|---|---|---|---|---|
| one step, `float` (the ceiling) | 20.0 | 19.2 | 628M | 605M |
| one step, `Raw[str]` | **41.4** | **21.8** | 3.20M | 4.57M |
| one step, semantic `str` | 55.2 | 38.6 | 0.26M | 0.25M |
| one step, `float` + `float` param | 21.7 | 21.0 | 611M | 607M |
| one step, `Raw[str]` + `Raw[str]` param | n/a | **22.9** | n/a | 4.40M |
| one step, `Raw[bytes]` span | **67.8** | **23.7** | 16.2M | 17.8M |
| branch condition, `float` (the ceiling) | 22.7 | 22.3 | 315M | 317M |
| branch condition, `Raw[str]` | **50.0** | **30.2** | 3.11M | 4.50M |
| branch condition, semantic `str` | 137.4 | 112.7 | 0.15M | 0.15M |

`n/a`: a `Raw[str]` param raised `PydanticSchemaGenerationError` before.

And the string-gated tree from `benchmarks/tree_walk.py`, `score()` p50, against the
numeric tree in the same run:

| | before | after |
|---|---|---|
| gated tree, fused | 179.4 | **96.2** |
| gated tree, stepped | 166.2 | **84.4** |
| numeric tree, fused | 68.7 | 68.3 |

A `Raw[str]` single record now costs 2.6 µs over a pure numeric kernel, down from 21.4;
with a tunable `Raw[str]` param it is 1.9 µs over the same step with a `float` param.
Batch throughput went up too (3.2M to 4.6M rows/s), because the old two-pass `encode`
was the batch cost as well. Semantic `str` never changed but drops 17 µs from the
hoisting alone, and the string-gated tree is within 16 µs of the numeric one.

What is left is the object-dtype block a `str` value needs in the record,
`State.representation`'s cache entry, and one `dict.get` plus one `np.fromiter` per
value. Each is under a microsecond; nothing left is worth a special case for `n == 1`.

## `Raw[str]` conditions do pack

A `branch` condition reading a semantic `str` cannot join the fused branch kernel
(`tests/control_flow/test_arity.py::test_a_condition_reading_a_str_input_is_right_but_does_not_pack`).
The same condition annotated `Raw[str]` packs: `exe.runner.packed` holds the branch,
no warning fires, and it runs 3.7x faster on a single record (30.2 µs against 112.7)
and 30x faster in batch (4.50M rows/s against 0.15M). That is the answer for a user
who needs the old packing speed, and it works.

## The global counter in `raw_str()`

`raw_str()` assigns codes from a process-global counter in call order, so two
processes with different import orders disagree on the code for the same string, and
a code read from a step body is frozen into the machine code as a numba constant.
Probed across processes (`NUMBA_CACHE_DIR` set, the same step compiled with extra
constants registered first to shift the counter):

| registered first | code | fingerprint | cache hits | misses | answer |
|---|---|---|---|---|---|
| none | 0 | `47ce7b4e0caf` | 0 | 1 | right |
| none | 0 | `47ce7b4e0caf` | 1 | 0 | right |
| three | 3 | `556cf87a8948` | 0 | 1 | right |
| three | 3 | `556cf87a8948` | 1 | 0 | right |
| none | 0 | `47ce7b4e0caf` | 1 | 0 | right |

So it is safe today: `fingerprint` hashes the values of the globals a step reads, and
`_SaltedCache` puts that in the disk-cache key, so a code change is a cache miss, not
a stale hit. The cost is one recompile per distinct import order, and one cache entry
each; both orders keep working.

### A content-derived code would be worse, not better

Deriving the code from the string would make it stable across processes. It is slower
and not safe at int32 width:

| encoder | one value | 200k values |
|---|---|---|
| shared counter table (`string_code`) | 0.95 µs | 7.57M rows/s |
| `zlib.crc32` masked to 31 bits | 1.53 µs | 3.05M rows/s |
| `blake2s(digest_size=4)` masked to 31 bits | 2.66 µs | 0.65M rows/s |

A table lookup beats hashing because CPython already interns and caches the string's
own hash, while a content hash must read every byte. And 31 bits is not enough room:
the birthday bound gives a 2.3% chance of a collision at 10,000 distinct values and
90% at 100,000, and `blake2s` truncated to four bytes really did collide 12 times over
200,000 distinct values. A collision is two different strings comparing equal inside a
kernel — a silent wrong decision, traded for a cache miss that costs a recompile.
Keep the counter.

### The two tables were a live bug, reachable through `Session.reload`

`StringCodes` kept its own table seeded from a copy of the global one and merged the
global back in on every `encode`, while assigning new codes from its own `len`. A
`raw_str()` call made after a run then renumbered a string the run had already coded:

```
run 1: {'walk-in': 1, 'broker': 2}
raw_str('broker') later -> 1
run 2: {'walk-in': 1, 'broker': 1}
```

`walk-in` and `broker` share code 1, so a step comparing against either matches both.

That arithmetic alone is not enough to reach a wrong answer: a plan is fixed at bind
time, and a fresh `bind` builds a fresh runner whose table starts from the global one,
so a constant named after one runner drifted cannot get into a step that runner runs.
**Hot reload closes the gap.** `Edits._apply` does `copy.copy(self._runner)`, a shallow
copy, so the reloaded runner shares the drifted `StringCodes` object, while the
reloaded step's constant comes from the global counter that is still behind. On
`d825f31`:

```
global table after run 1: {'priority': 0}
runner table after run 1: {'priority': 0, 'walk-in': 1, 'broker': 2}
raw_str('broker') -> 1
hit for ['walk-in', 'broker'] -> [True, True]      # expected [False, True]
```

A step comparing against `raw_str("broker")` matched `"walk-in"`. That is a silent
wrong answer through the public `Session.reload` / `Session.watch` API — the notebook
and `session_ws` path, and the normal way a `Raw[str]` pipeline is edited while it
runs. One table removes it. Guarded by
`tests/debug/test_session_reload.py::test_a_reloaded_step_matches_the_constant_it_names_against_values_the_run_already_coded`
and, for the table invariant alone,
`tests/run/test_compiled_fallback.py::test_a_constant_named_after_a_run_never_renumbers_a_value_it_saw`.

## A `Raw[str]` param, so the fast path stays tunable

`Raw[str]` could only be compared against a module-level `raw_str()` constant, which
made it useless for anything a params document tunes — and retuning a string literal
without recompiling is the point of a param. A param annotated `Raw[str]` now works:

```python
PRIVATE = raw_str("private")

def sector_rate(sector: Raw[str], private: Raw[str] = param("private"),
                rate: float = param(0.9, gt=0)) -> float:
    return rate if sector == private else 1.0
```

The document holds the string (`{"sector_rate": {"private": "government"}}`) and
`parameters().defaults()` shows a string; only the bundle the step receives holds the
code. The conversion is one `np.int32(string_code(value))` in `NodeParams.bundle`, so
it happens where the params cache already memoises per (document, node), and every
runner gets it — interpreted compares codes too, so the modes agree. `np.int32`
rather than a Python `int` keeps the kernel's signature at `int32` whatever the value,
so a retune never recompiles
(`tests/run/test_compiled_modes.py::test_retuning_a_raw_str_param_never_recompiles`).

The mismatch is now refused instead of answering `False` in silence: a node mixing
`Raw[str]` and plain `str` across its inputs and params raises a `WiringError` naming
both and saying which annotation to change. That check is in `resolve`, so it fires in
every mode, before anything runs.

## For the guide

A block to fold into `GUIDE.md` (single record first, one runnable example):

> ### Strings in a compiled kernel
>
> A `str` input is compared as a string, which no shared array kernel can hold, so the
> step runs compiled one call per row. When a string only ever gets compared for
> equality, annotate it `Raw[str]`: it enters the kernel as an int32 dictionary code
> and joins the fused kernel like a number, including inside a `branch` condition.
> `raw_str()` gives you the code of a literal, and a `Raw[str]` param is tuned as an
> ordinary string.
>
> ```python
> from decider import Raw, flow, param, raw_str
>
> def sector_rate(sector: Raw[str], private: Raw[str] = param("private"),
>                 rate: float = param(0.9, gt=0)) -> float:
>     return rate if sector == private else 1.0
>
> exe = Engine().bind(flow(sector_rate), mode="fused")
> exe.score({"sector": "private"})["sector_rate"]                                 # 0.9
> exe.score({"sector": "private"}, params={"sector_rate": {"private": "public"}})  # 1.0
> ```
>
> A code is only ever equal or not equal: `<`, `in`, `startswith` and anything else
> string-shaped need a plain `str` (or `Raw[bytes]`, which gives the kernel the UTF-8
> bytes). Codes are assigned per process in first-seen order, so they are not stable
> across processes and are not something to store or compare between runs.

## Left alone

`njit.py`'s `_probe_signature` still types a plain `str` param of a scalar node as
`int32`, which nothing converts. Nothing can reach a wrong answer through it any more:
a `Raw[str]` input beside a plain `str` param is a `WiringError`, and the remaining
case (a plain `str` param on a node with numeric inputs) costs one extra numba
specialisation on first use, because the probe's signature does not match the unicode
the bundle actually carries. Making the probe honest would flip which semantic-`str`
steps fall back to Python, which is a bigger blast radius than the one compile it
saves, so it stays as it is.

A `Raw[str]` param on a *row* node (a tree or table config) would reach `typeof(bundle)`
as a string in the probe and as `np.int32` at run time: one extra specialisation, not a
wrong answer. No config declares one today, so there is nothing to substitute for.
