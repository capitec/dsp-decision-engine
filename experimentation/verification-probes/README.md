# Verification probes

Two claims from the overnight string/type strands were load-bearing enough to
check directly rather than accept from the strand that produced them. These are
those checks, standalone and reproducible.

```
../../.venv/bin/python probe_decider2_refcount.py
../../.venv/bin/python probe_category_mask.py
```

## `probe_decider2_refcount.py`

`experimentation/numba-c-string-matching/` found that writing a tree's row body
as a *separate* numba function taking arrays costs ~365 ns/row, because numba
emits NRT reference-counting traffic for every array argument on every row. That
is roughly the shape of decider2's own `trees/interpreter.py:walk_tree`, so the
question is whether decider2 pays it today.

It does not:

```
A  decider2 shape (per-row call into walk_tree, 7 array args) : 45.64 ns/row  NRT (4, 11)
B  same body inlined in one loop, arrays captured             : 43.80 ns/row  NRT (0, 7)
ratio A/B = 1.04x
```

decider2's structure arrays are loop-invariant, so the bookkeeping hoists out of
the loop. The trap stays live for a **typed** walker, where per-row feature
arrays would be passed in and are not invariant — keep per-row inputs as scalars
or raw integers.

## `probe_category_mask.py`

Both string strands recommend the per-category mask (run the pattern over the
distinct values once, then one array lookup per row). This measures it end to
end on 1M rows over 12 distinct values, including the costs the headline figure
leaves out:

```
per-row mask lookup in kernel            :  0.94 ns/row
total, column already dictionary-encoded :  1.07 ns/row
polars str.contains over every row       : 34.09 ns/row
total, if you must encode it yourself    : 50.53 ns/row
```

The 32x win holds **only** while the column arrives already encoded — which
decider2 does at the boundary today, and which is the precondition to re-check
before adopting the mask.

## `run3.sh` + `compile_cost.py` — the inline/typed attribution

The typed-features strand reported a tree getting 1.6–1.8x *faster*, which a
type discriminator should not do. It had changed two things at once: the typed
representation, and `inline="always"` on `walk_tree`/`path_fn`. These scripts
measure the third cell the strand never ran — the **old, untyped code with only
the inlining change** — interleaved with the other two on the same box.

Setup: extract `git archive f90488c` into `base/`, copy it to `base_inline/`,
and in that copy change `interpreter.py`'s `@njit(cache=True)` on `walk_tree`
to `@njit(inline="always")` and `encode.py`'s cached `path_fn` to
`@njit(cache=True, inline="always")`. Copy the strand's
`decider2/evaluation/typed-features/bench_typed_features.py` next to them as
`bench.py`. Then `./run3.sh`.

Result, mixed tree (10 Float64 + 4 Int64 + 2 Boolean), 200k rows, two rounds,
byte-identical output digests across all three:

| | apply() end to end | tree walk alone |
|---|---|---|
| f90488c as it stands | 247–262 ns/row | 201–217 ns/row |
| f90488c + inlining only | **125–133** | **96** |
| typed features (inlining included) | 136–154 | 117–151 |

The whole speedup is the inlining, not the typed split; the typed
representation costs ~12–35% against it, consistent with the numba+C strand's
independent 5–20% for the same discriminator.

`compile_cost.py <label>` measures the build-time side, with
`NUMBA_CACHE_DIR` set and `NUMBA_DEBUG_CACHE=1` to count cache saves/loads:

```
today               cold 1.97 s  warm 0.89 s   8 saved /  5 loaded
today + inlining    cold 1.61 s  warm 1.05 s   6 saved /  6 loaded
typed features      cold 2.83 s  warm 1.55 s  14 saved / 14 loaded
```

Inlining costs nothing to compile and caches cleanly; the typed split costs
~1.4x cold and ~1.5x warm start.

## `probe_int64_fix.py` — how much of the int64 fix is opt-in

The typed-features branch fixes the silent wrong answer above 2^53. This checks
end to end, through the real pipeline in all three execution modes, how much of
it applies without a declaration:

```
                                             today        the branch
  no declaration (every existing document)    [1, 1]  wrong   [1, 1]  still wrong
  declared feature_types={'n': int}           no API          [1, 0]  correct
```

Run it with `PYTHONPATH` pointed at each tree in turn. The opt-in behaviour is
deliberate and pinned by a test in the branch — silently retyping existing
documents would change answers nobody asked to change — but it means the bug
stays live in every tree that does not declare.

## `probe_arrow_zerocopy.py` — which polars accessor copies a String column

The overnight strands all concluded "polars string buffers are not zero-copy",
measured through `Series._get_buffers()`. The Arrow strand then found the
memory *is* reachable without a copy, through the Arrow C Data Interface. This
settles it with two witnesses that need no knowledge of the layout — cost
scaling, and resident memory:

```
      rows   arrow_c_stream     _get_buffers    ns/row (stream)  ns/row (buffers)
   100,000            1.2us         1469.8us            0.012             14.7
 1,000,000            1.3us        18925.0us            0.001             18.9
 4,000,000            1.2us        77526.1us            0.000             19.4

4,000,000 rows x 19 bytes = ~76 MB of string data
  after 5x __arrow_c_stream__()     : delta   +0.0 MB
  after 5x _get_buffers()           : delta +222.0 MB   (~44 MB per call)
```

`__arrow_c_stream__()` is flat at ~1.2 us however many rows there are and adds
no resident memory. `_get_buffers()` is O(rows) and allocates a fresh copy every
call, because it converts binview to a `large_utf8` offsets+values pair.

So the accessor was the copy, not the memory. Anything reading polars strings
from a kernel should take the C Data Interface route.
