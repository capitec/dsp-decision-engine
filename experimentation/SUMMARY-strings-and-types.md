# Strings and feature types in decider2 — where we got to

*Written for someone who has not been following the detail. Read the first two
sections and stop, unless you want the evidence.*

---

## 1. The two questions

**Can a decision tree match a string pattern — a prefix, a regex — rather than
only checking equality?** Today it cannot. `UnaryStringMatch` is restricted to
`match_type="exact"`; prefix, suffix, substring and regex all raise
`UnsupportedInKernel`.

**Can a tree hold floats, ints, bools and strings honestly?** Today it cannot.
Every feature is coerced into one `float64` array.

They turned out to be the same question, which is why they were worked together.

---

## 2. Why this matters, in one example

Everything funnels through one `float64` array. So:

    two DISTINCT int64 values, 9007199254740992 and 9007199254740993,
    compared `== 9007199254740992`
        got      [match, match]
        expected [match, no match]

`float64` cannot represent both, so they collapse to the same number and both
match. Doc 03 §1.2 mandates **scaled int64 for money precisely to avoid float
error**. So the design's own money rule is broken by its own feature array —
silently, with a plausible answer, which doc 03 §2.1 calls the worst failure
mode the design can have.

In practice 2^53 cents is about 90 trillion rand, so no real loan reaches it.
But "no realistic input reaches it" is exactly how the `5 < x < 10` bug and the
16-feature ceiling looked, right up until they didn't.

The same single array is why a category code (a small integer) rides in a
`float64` slot, and why nothing can detect that `<` on a categorical feature is
meaningless.

---

## 3. What was actually tried

Four independent strands, run overnight, each measured rather than argued. The
fourth was not in the original plan — the C work turned it up (section 5):

| strand | question | status |
|---|---|---|
| **Rust** | match strings in Rust, called from inside the compiled kernel | **done** |
| **Numba + C** | same, but C instead of Rust — no second language in the build | **done** |
| **Typed features** | split the one array into float64 / int64 / bool / codes | **done** |
| **Pure numba bytes** | prefix/suffix/substring over raw bytes, no C and no Rust | *running* |

---

## 4. Rust — built, measured, and NOT recommended

### What it is

A small Rust library (no Python bindings) exposing plain C functions. decider2's
compiled kernel calls them directly, passing the string bytes as raw pointers.

### The numbers that decide it

Low-cardinality column — 12 distinct employers, which is what real credit data
looks like for `sector`, `product_code`, `employer_type`:

| strategy | ns per row |
|---|---|
| **per-category mask** (no Rust at all) | **0.8 – 1.8** |
| frame-tier precompute (today's answer) | ~35 |
| lazy Rust at the node | 16 – 45 |

**The per-category mask beats every Rust strategy at every selectivity**, and it
needs no Rust: run the regex over the 12 distinct values in Python, once per
batch, then the kernel does a single array lookup per row.

High-cardinality column — a distinct string per row, e.g. merchant descriptors:

| rows reaching the regex node | 0% | 1% | 50% | 100% |
|---|---|---|---|---|
| lazy Rust, ns/row | 6.7 | 7.8 | 33.0 | 44.8 |
| frame tier, ns/row | ~35 | ~35 | ~35 | ~35 |

Here lazy Rust **does** win — 4.5x at 1% selectivity, crossing over around 75%.
That is the case you predicted, and it is real.

### Why it is still not worth adopting

The entire prize is one frame-tier pass per string node per batch: ~35 ns/row,
or **35 ms per million rows**. A quarter of that is polars' own buffer
conversion, which Rust does not avoid. Against that: a second language, a wheel
per platform, and a new way to crash.

### Two things I had told you that were wrong

**"polars string buffers are zero-copy, there is nothing to pack."** False on
polars 1.41. A `String` column is stored as `Utf8View`, so asking for the
offsets/values buffers *materialises* them every call — verified independently:
two calls return different memory, at 6.8 ns/row for 200k rows. At 0%
selectivity that conversion IS the whole cost of the lazy path.

**"the Rust regex crate is 25.6 ns/call."** It is ~40 ns on the 13–22 byte
strings that actually occur here, and pattern complexity barely matters
(`^dog`, `(?i)^dog`, `dog` all within ±5 ns). Fixed per-call overhead dominates
short haystacks.

### Checked independently — and the mask has a condition attached

Both strands recommend the per-category mask, so I measured it myself, end to
end, on a million rows over 12 distinct employers:

| | ns per row |
|---|---|
| the lookup itself, inside the kernel | **0.94** |
| building the mask from categories already in hand | 0.05 |
| getting the int32 codes out | 0.03 |
| **total, if the column already arrives dictionary-encoded** | **1.07** |
| polars `str.contains` over every row — today's answer | 34.09 |
| **total, if you must dictionary-encode the column yourself** | **50.53** |

So the mask is a **32× win — but only on the condition that the column arrives
already encoded.** If it does not, `unique()` plus the encode costs 47 ns/row
and the mask *loses* to the frame tier it was meant to replace.

decider2 satisfies the condition today: string columns are dictionary-encoded
into int32 codes at the boundary, which is exactly why `match_type='exact'`
works at all. The mask rides on work already paid for. That is the whole reason
this is the right answer — and it is also the thing to re-check before adopting
it, because if the encoding step is ever moved or made lazy, the recommendation
inverts.

### What it did prove, and is worth keeping

**Caching works.** Historically, calling out to a C or Rust symbol destroyed
numba's on-disk compilation cache — every process would recompile. The fix is
precise: pass the function pointer as an *argument*, never capture it as a
global. Verified cold and warm in two separate processes, zero re-saves on the
warm run.

**Panic safety is real but fragile.** A protected call returns an error code and
the process carries on. The unprotected twin **aborts the whole process**
(`SIGABRT`). Two separate disciplines have to hold, and nothing enforces either
— which in a bank's credit path is a worse failure than an exception.

---

## 5. Numba + C — built, measured, and NOT recommended for speed

### What it is

A 230-line C file wrapping the system PCRE2 regex library. The compiled numba
kernel calls it directly. **C owns the table of compiled patterns**; the kernel
only ever passes a small integer id, which C bounds-checks before touching
anything. An earlier version handed raw pointers to the kernel, and a made-up
pointer segfaulted — moving the table into C closed that at no measurable cost.

### The finding that overturns an earlier conclusion

An earlier experiment (EXPERIMENTS.md §T) tried C regex from a kernel, found it
slower than polars, and concluded C was the wrong route. **That was a verdict on
the wrong library.** It used glibc's POSIX `regexec`. Same harness, same data:

| regex engine, called per row from inside the kernel | ns per call |
|---|---|
| **PCRE2 with JIT** | **29 – 40** |
| polars' own Rust regex, for comparison | 24 – 46 |
| PCRE2 without JIT | 74 – 108 |
| glibc `regexec` — what §T measured | 71 – 248 |
| Python's `re`, for scale | 622 – 641 |

So C is not the slow part; glibc was. PCRE2-JIT from a kernel is level with
polars' Rust engine, and the Rust strand's engine is level with both. **Engine
choice no longer turns on speed** — all the credible options are 30–40 ns.

### Does lazy evaluation pay off?

This is the question you raised: the frame tier runs the regex over every row,
while a tree short-circuits, so in `ft1 > 5 AND ft2 > 10 AND ft3 ~ /^dog/` where
nothing passes `ft1 > 5`, a lazy regex at the node should run **zero** times.

It does, and it wins where you said it would. One million rows, milliseconds per
batch, each strategy carrying its own setup cost:

| rows reaching the regex node | today (frame tier) | lazy C at the node |
|---|---|---|
| 0% | 33 – 62 ms | **15 – 24 ms** |
| 1% | 39 – 62 ms | **19 – 26 ms** |
| 50% | 56 – 83 ms | 56 – 62 ms |
| 100% | 46 – 81 ms | 62 – 79 ms |

2.5–3.5× at the selective end, crossing over around half the rows. The mechanism
is sound and the prediction was right.

### Why it is still not recommended

Convert that to per row: the whole win is **20–40 nanoseconds a row**. On a
10,000-row scoring batch, 0.2–0.4 milliseconds. Against that you take on a C
build step, a system library to vendor per platform, a raw-pointer surface, and
the hazard below — in a bank's credit path.

### The hazard that decides it

PCRE2 is a *backtracking* engine. A rule author writing a pattern like
`^(a+)+$` — which looks entirely innocent — makes it explore exponentially:

- **211 microseconds per row** with the safety limit the shim sets
- **42 milliseconds per row** at PCRE2's own default limit

That is a denial-of-service axis created by a rule change, not a code change. It
cannot happen with polars' regex or Rust's, which are linear-time by
construction and have no pathological inputs. If in-kernel regex is ever
adopted, **this is the axis to choose the engine on** — and it points at Rust,
not at PCRE2.

Two smaller ones: the shim keeps one scratch block per pattern, so it is not
safe under parallel rows (`prange`) without per-thread state; and it links the
system `libpcre2`, which is not a Windows story.

### The part that IS worth taking

`^dog` is not a regex problem — it is a prefix test. A ten-line `memcmp`
matcher does it in **9–14 ns**, a third of any regex engine, and cannot
backtrack because there is nothing to backtrack. Prefix, suffix and substring
are the same. That suggests the genuinely attractive move: lift
`match_type='prefix'|'suffix'|'substring'` out of `UnsupportedInKernel` with
plain byte comparison — and it may not need C at all, since numba can compare
bytes in a `uint8` array itself. That is a couple of hours to find out, and it
would remove most of the restriction with no dependency and no new failure mode.

### Checked independently

**Caching holds.** I read numba's own logs rather than the summary: cold run
saved 2 kernels and loaded 0; warm run, separate process, saved 0 and loaded 2.
In that same warm process a deliberately-wrong twin that captures the C symbol
as a global produced numba's refusal — *"Cannot cache compiled function … it
uses dynamic globals (such as ctypes pointers …)"*. The rule from §W is exact
and now demonstrated twice: **pass the pointer as an argument; never capture it
as a global.**

**The int64 precision failure is real.** Reproduced directly: values
`[2⁵³+1, 2⁵³, 2⁵³−1, 2⁵³+1]` against threshold `2⁵³` give `[0,1,0,0]` for `==`
as int64 and `[1,1,0,1]` as float64 — the wrong answer in two of four rows, and
`>` is wrong in all four. This is §2's example, confirmed.

**One alarm was raised that does not apply to decider2.** The strand found that
writing the row body as a *separate* numba function taking arrays costs
**365 ns/row** — numba emits reference-counting traffic for every array
argument, on every row. Since that is roughly the shape of decider2's own
`walk_tree`, I measured decider2's real walker directly rather than take it on
trust:

    decider2 today (per-row call into walk_tree, 7 array arguments) : 45.6 ns/row
    same body inlined into one loop, no per-row call                : 43.8 ns/row
    ratio 1.04x

**decider2 does not pay this today** — its arrays are loop-invariant, so the
compiler hoists the bookkeeping out of the loop. But the trap is live for the
typed split, where per-row *feature* arrays would be passed in and are not
invariant. It is a design constraint on section 6's work, not a bug to fix now.

**`_get_buffers()` costs 11 ns/row here**, consistent with the 6.8 ns/row I
measured myself. Both strands now independently contradict what I told you
earlier: it is not zero-copy. At 0% selectivity that conversion is essentially
the entire cost of the lazy path.

Full detail: `experimentation/numba-c-string-matching/RESULTS.md`; every
measurement in `results.jsonl`; `./run_all.sh` reproduces in about eight
minutes.

---

## 6. Typed feature arrays — built, green, and the headline needs unpicking

### What it does

Every feature a tree reads is no longer coerced into one `float64` array. A row
is now six typed arrays — float64, int64, bool, category code, string spans,
string bytes — and each node in the tree carries a small integer saying which
one its feature lives in. That integer also picks which threshold tuple to
compare against, so an int64 feature is compared against an int64 threshold.

The int64 case from section 2 is now right:
`[9007199254740992, 9007199254740993] == 9007199254740992` gives `[1, 0]`
instead of `[1, 1]`, in every execution mode and in `score()`.

Knowing a feature's type at build time also lets the encoder **refuse** things
the single array could never see. `sector < 5` on a string column used to
compile happily and compare dictionary codes — numbers with no order and no
meaning. It is now a build error that names the tree, the node and the feature.
So are: a threshold on a boolean, `is_true` on a string, `string_match` on a
number, and a fractional threshold like `5000.5` on an integer feature.

Existing documents encode byte-identically; the wire format is untouched;
retuning an integer threshold still triggers zero recompiles; the 400-feature
tree still builds. I ran the suite myself in the branch: **563 passed**.

### One thing to be clear about: the fix is opt-in

A feature only gets a real type if the tree **declares** one. Undeclared, it is
still inferred `float`, exactly as today. I checked this end to end rather than
read it:

| | today | the branch |
|---|---|---|
| no declaration — *every document that exists* | `[1, 1]` wrong | `[1, 1]` **still wrong** |
| declared `feature_types={'n': int}` | no such API | `[1, 0]` correct |

So merging this does not fix the money trees you already have; it gives them a
way to be fixed, one declaration at a time. The strand pins that behaviour with
a deliberate negative-control test, which is the right call — silently
retyping every existing document would change answers no one asked to change —
but it means "the int64 bug is fixed" is only true of trees that opt in. If
that is not what you want, the follow-up is inferring the kind from the
*column's* dtype at the boundary rather than from the declaration, which is a
bigger and more invasive change than this branch makes.

### The headline says 1.6–1.8× faster. That is not the typed split.

The strand reports the tree getting *faster* — 249–255 ns/row down to
140–156. That was surprising, because a type discriminator per node should
cost a little, not pay for itself. It turned out the strand changed two things
at once: the typed representation, **and** marking the walker
`inline="always"` so it compiles into the per-row loop instead of being called.

So I measured the third option it never ran — the **old, untyped code with only
the inlining change** — interleaved with the other two, twice, on the same box.
All three produce byte-identical output:

| mixed tree, 200k rows | end to end | the tree walk alone |
|---|---|---|
| today | 247 – 262 ns/row | 201 – 217 ns/row |
| **today + inlining only (a two-line change)** | **125 – 133** | **96** |
| typed features (inlining included) | 136 – 154 | 117 – 151 |

**The entire speedup is the inlining.** It is available now, on the current
code, without the typed split: roughly **2× on the tree path** from marking two
functions inline. The typed representation, measured against that, *costs*
about 12–35% — which is what it was always expected to cost, and matches the
Numba+C strand's independent 5–20% for the same discriminator.

This does not make the typed work wrong; it makes the case for it honest. It is
a **correctness** change that costs roughly a fifth of the walk, not a
performance change. That is a much easier thing to decide about.

### The finding underneath, which is worth more than either

The strand's first typed attempt was **2× slower** (441–464 ns/row), and
chasing that turned up the real rule: **numba arrays crossing a genuine call
boundary, per row, are expensive.** Each array is seven scalars; six of them
plus the tree's own arrays is around ninety scalars pushed and reloaded on
every row. One float64 array crossed that boundary nearly free, which is why
nothing ever noticed. Inlining removes the boundary, so the row buffers stay in
registers and the tree's arrays become compile-time constants.

That is the same mechanism the Numba+C strand hit from a different direction
(its 365 ns/row refcount trap), and the same one my own probe found does *not*
bite decider2 today. Three independent encounters with one rule:

> **Do not put a non-inlined layer between the per-row kernel and the walker.**

It is now documented at both call sites in the strand's branch.

### Compile cost, since inlining usually has one

Measured cold and warm, fresh cache, separate processes:

| | cold build | warm start | caches cleanly |
|---|---|---|---|
| today | 1.97 s | 0.89 s | yes |
| today + inlining | **1.61 s** | 1.05 s | yes — 6 saved cold, 6 loaded warm, 0 re-saved |
| typed features | 2.83 s | 1.55 s | yes — 14 saved cold, 14 loaded warm, 0 re-saved |

Inlining costs nothing at build time. Typed features cost about 1.4× the cold
compile and 1.5× the warm start, because there are more specialisations to
load — a startup cost, which you said you can work around.

### Caveats worth knowing before merging

- A computed expression (`x - y > 10`) still cannot read an int-, bool- or
  string-typed feature — arithmetic there is float64. It is a **loud error**,
  not a silent widening, but it means money arithmetic inside an expression
  needs the feature declared `float`.
- A Float64 column handed to an int-declared feature is still truncated at the
  boundary, the same rule every hand-written `-> int` step already lives under.
- The raw-string slot is **reserved and driver-tested but not wired**: a
  `bytes`-annotated input can carry polars' offsets and bytes into the kernel
  zero-copy, and that is verified, but the boundary does not yet produce such a
  column and the walker has no node kind that reads it. That is the hook the
  string work would use.
- Pre-existing and untouched: `expr.py`'s per-node closures are `cache=True`
  while capturing other compiled functions — the pattern we established must
  not be cached. It predates this work; flagged, not fixed.

The branch is `worktree-agent-a47fc4a23192c7421`, uncommitted, with its own
write-up at `TYPED_FEATURES.md`.

---

*The combined recommendation, and the fourth strand (pure-numba byte matching,
no C at all), to follow.*
