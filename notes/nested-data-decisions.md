# Nested data: what we decided, and the experiments that decided it

`notes/tmcfeval/DECIDER_GAPS.md` §3 ("Lists of records have no grain of their
own") asks for two things: per-item rules as ordinary steps, and parent logic
over a row's items in a kernel. Its user-feedback block adds a third: a list of
records should *just work* undecorated, and an opted-in spelling should compile
well.

Three spikes answered it. Each has its own note with the full measurements;
this file records what was decided and why. All figures are from a loaded
shared box: read the ratios, and the control row in each note.

## Where it started

Measured on the merged branch before any of this:

- `list[dict]` and `list[Item]` (a `TypedDict`) already just work in every mode,
  including `for i in items`, returning an item or `None`, and returning a list.
  The first half of the user's feedback was already satisfied.
- `Rows[Item]` existed but **lost to plain `list[dict]` on a light body** at
  every size, because `build_rows` assembled the arrays with a Python loop.
- A `loop` whose body read `items: Rows[Item]` did **not** pack:
  `packed loops: []`. §3's central performance claim was unrealised.
- An `Item` field could only be `float`, `int`, `bool` or `float | None`.

## Decision 1: read `Rows[Item]` from Arrow — done

`notes/nested-rows.md`. The Arrow layout for `list<struct<...>>` already *is*
flat per-field arrays plus offsets, so the arrays are views, not copies.
Crossover measured at 256 rows, below which the Python read wins, so `score()`
is Python by construction. 2.2x in bulk, and the fixed per-call cost fell from
19 to 9 us.

## Decision 2: a `Rows[Item]` step joins the shared array kernel — ship

`notes/nested-rows-in-kernel.md`. Pass the flat arrays plus per-row `lo`/`hi`
as ordinary kernel arguments and slice them **inside** the kernel, so the
ragged shape is never a Python-side value.

| bundle search, 9 items, 511 masks | before | after |
|---|---|---|
| fused `score()`, one order | 60.58 ms | **0.114 ms** |
| fused `run()`, 100 orders | 576 ms | **3.45 ms** |
| packed loops | `[]` | `['order/search']` |

Verified independently, same answer both ways. §3's "119 ms -> 1.4 ms" holds;
the ratio is larger and the absolute numbers differ because its body was
heavier. It lands within 10x of a hand-written `@njit` search at this size, and
1.2x at 16 items, where the residual is the fixed cost of a `score()` call.

Why this shape: **no new numba type was needed.** A `Rows[Item]` value already
is a numba type, and the missing piece was only constructing one inside a
kernel. The version-to-many-columns layout change we expected was avoided —
`lo`/`hi` per row rather than an `offsets[n+1]` makes a branch or loop row
subset two fancy indexes with no child copy. Single-record improves rather than
regressing, because what disappears is building a namedtuple of views per row.

Known limits, kept deliberately: `Rows[Item] | None` cannot join a kernel (one
signature, no in-band `None`) and keeps the per-row path; two different `Item`
types over one column raise.

## Decision 3: an `Item` field may be a `str` — done

`notes/nested-item-fields.md`. A `list<struct<... utf8 ...>>` child holds its
strings in exactly the buffers a span points into, and spans now have real
string operations, so the field is an array of spans: `types.Array(SPAN, 1, "C")`
works with no numba glue, because a span's data model is already `[2 x i64]`.
**No string is copied**: +43-97 ns an item, against 64 ns for a float field.
`==`, `len`, `startswith`, `endswith` and `in` are exact against CPython over
ASCII and multi-byte UTF-8. One value serves every mode, which is what keeps
`assert_equivalent` able to check it.

## Open: the physical shape (namedtuple of arrays vs a record per row)

`notes/nested-item-fields.md` recommends a record array per parent row: it makes
`for i in items:` and `i.el_1` work, it is a strict superset of today's API so
nothing breaks, and it measured 2-5x cheaper per call because numba unboxes
**per namedtuple member**, charging a step for fields its body never reads.

That performance argument was measured against the *per-row dispatcher*, which
decision 2 removes. So the question was handed back with a measurement attached:
does a record still pay once the step is in the kernel, or is the remaining case
purely ergonomic? Until that answer lands, the namedtuple stays.

Two things that argument settled either way: today's `for i in items:` over the
namedtuple is a **trap** (with an all-same-dtype `Item` it compiles and
silently iterates the *fields*), and a step cannot return an item — `-> Item`
and `-> dict` raise, so the honest spelling is to return the index.

## Held: `each`, a child grain

`notes/child-grain.md`. Buildable for ~60 lines with **no engine change**: give
the child its own `State`, since `State` is keyed by one plan's version ids, and
the parent stores only the gathered list column. It is ~8x faster than today's
`frame_step` workaround on a single record (184 us vs 1463) and **6-10x slower
in a batch**, because polars is right for 30k items in one column and wrong for
three. That tradeoff is a product call, not a technical one, so it waits.

What it would not buy for free: params and sessions at the child path.
`parameters()` is `{}` for the child and `break_at("order/clean/heavy")` matches
nothing. The blocker is not `State` but `Checkpoint`, which is
`(origin, when, arm, iteration)` with no coordinate for *which item*.

## Rejected

- **A helper making `missing_as` easy inside a `frame_step`**, as an alternative
  to `each`. Built and measured: 1598 us against 1463 for the hand-rolled
  version. It removes the ugly code and none of the cost.
- **Iteration via the namedtuple.** Not possible, and it silently does the wrong
  thing today (above).
- **A record array through `np.recarray`.** Slicing one costs 5.6 us against
  0.24 us for a plain record view — unusable per row.

## Findings that outlived the question

- **One no-op `frame_step` takes `score()` from 27.4 us to 248.2 us** — a 9x
  penalty on the single-record path, for any pipeline with a frame step, and the
  guide currently recommends one for per-item logic. Same root cause as `run()`
  on a one-row frame costing 153 us against `score()`'s 30. This is why `each`
  is a speedup rather than a cost, and it is worth fixing on its own.
- Every representation annotation had a path where the engine quietly delivered
  a different representation than the annotation promised. Eight were found and
  fixed across these rounds; the ones from this batch are in
  `notes/strings-span-ops.md` and the commit log. The rule that survived: a
  `Raw[...]` or `Rows[...]` annotation is a contract that holds even when the
  step runs in Python, while `bytes` is a contract for a scalar step only — a
  row node falling back matches on the strings themselves.
- Latent, and would bite the moment a child frame exists: `_concrete` maps
  `Null -> String` inside nested types, so a `List(Struct({price: Float64,
  weight: Null}))` column becomes `...weight: String`. `Rows[Item]` survives it
  by reading Python values.
- `corpus()` gives any list input a `Float64` column of `1.0`, so no
  empty-list, null-list or many-item case exists in the generated corpus.
- `serving/parse.py`'s `dummy` sends `1.0` for a `Rows[...]` input, so such a
  pipeline can only be warmed from a real `sample_request.json`.
