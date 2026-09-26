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

## Before and after

Same box, same script, interleaved rounds, best p50 of four (`mode="fused"`,
200k rows for the batch column):

| variant | p50 before | p50 after | rows/s before | rows/s after |
|---|---|---|---|---|
| one step, `float` (the ceiling) | 20.3 | 19.8 | 627M | 609M |
| one step, `Raw[str]` | **39.9** | **22.0** | 3.30M | 4.52M |
| one step, semantic `str` | 55.4 | 39.1 | 0.26M | 0.25M |
| branch condition, `float` (the ceiling) | 22.5 | 22.8 | 313M | 333M |
| branch condition, `Raw[str]` | **50.8** | **27.9** | 3.37M | 4.44M |
| branch condition, semantic `str` | 138.6 | 119.0 | 0.15M | 0.15M |

A `Raw[str]` single record now costs 2.2 µs over a pure numeric kernel, down from
19.6. Batch throughput went up too (3.3M to 4.5M rows/s), because the old two-pass
`encode` was the batch cost as well. Semantic `str` shares the hoisting and drops
16 µs without any change of its own.

The remaining 2.2 µs (5.1 µs in the packed branch, which reads two more columns) is
the object-dtype block a `str` value needs in the record, `State.representation`'s
cache entry, and one `dict.get` plus one `np.fromiter` per value. Each is under a
microsecond; nothing left is worth a special case for `n == 1`.

## `Raw[str]` conditions do pack

A `branch` condition reading a semantic `str` cannot join the fused branch kernel
(`tests/control_flow/test_arity.py::test_a_condition_reading_a_str_input_is_right_but_does_not_pack`).
The same condition annotated `Raw[str]` packs: `exe.runner.packed` holds the branch,
no warning fires, and it runs 4.3x faster on a single record (27.9 µs against 119.0)
and 30x faster in batch. That is the answer for a user who needs the old packing
speed, and it works.

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

### The two tables were a real bug

`StringCodes` kept its own table seeded from a copy of the global one and merged the
global back in on every `encode`, while assigning new codes from its own `len`. A
`raw_str()` call made after a run then renumbered a string the run had already coded:

```
run 1: {'walk-in': 1, 'broker': 2}
raw_str('broker') later -> 1
run 2: {'walk-in': 1, 'broker': 1}
```

`walk-in` and `broker` share code 1, so a step comparing against either matches both.
One table removes it;
`tests/run/test_compiled_fallback.py::test_a_constant_named_after_a_run_never_renumbers_a_value_it_saw`
guards it.

## Left alone

A scalar step that reads a `Raw[str]` input and compares it to a semantic `str` param
returns `False` for every row, in every mode, with no warning: the input is a code and
the param is a string. `njit.py`'s `_probe_signature` types such a param as `int32`,
which nothing converts — the conversion branch that would have done it was
unreachable, and is now deleted. Making the kernel probe honest (`unicode_type`) would
turn the step into a Python fallback that answers correctly in `stepped`/`fused` while
`interpreted` still answers `False`, which trades a consistent wrong answer for an
inconsistent one. It wants a decision: refuse the combination at wiring time, or give
`raw_str` a param form.
