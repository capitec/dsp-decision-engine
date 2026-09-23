# Strings and feature types in decider2 — where we got to

*Written for someone who has not been following the detail. The box below is
the whole answer; everything after it is the evidence, in the order it was
found.*

---

## The short version

| | verdict |
|---|---|
| **Inline the walkers** — ready on a branch, 1.6–1.9× on trees and 22% on Branch/Loop, identical answers | **take it** |
| **Strings straight from polars via Arrow** — no dictionary encoding, `exact`/prefix/suffix/contains at the node, pure numba, and *cheaper* than today | **take it — this is the string answer** |
| **Typed feature arrays** — fixes a silent wrong answer on money columns, costs ~20% of the walk | **merge, as a correctness change** |
| **Rust for string matching** — works, caches, 4.5× on a narrow case worth 35 ms per million rows | **no** |
| **C (PCRE2) for string matching** — works, caches, 2.5–3.5× on a case worth 20–40 ns/row, and it backtracks | **no** |
| **polars Rust plugin** — type-honest and 7–85× on big batches, but 4× over spec on single records, 252 crates, and a process-abort failure mode | **no, unless big batches become the main case** |

**If 90% of calls are single-record, almost none of the batch numbers below
decide anything.** At one row every engine is ~1 µs of actual work wrapped in
230–600 µs of plumbing. That is section 9's finding and it is the same wall
decider2 already hit — the fix is doc 05 §3.1b's whole-row marshalling, in
whatever language.

Two surprises. The typed-features branch reports itself 1.6–1.8× *faster*; it
is not the typed split — it changed two things at once, and **all** of the
speedup is the inlining, which you can have on its own today (section 6). And
the Arrow route you asked about, which you expected to cost a little more for a
simpler pattern, turns out to be **cheaper** as well as simpler (section 8) —
because the accessor every earlier strand used was the thing doing the copying.

Three things I had told you earlier are wrong, and are corrected in place with
the measurements: the Rust regex crate is ~40 ns not 25.6; "C regex is too
slow" was a verdict on glibc rather than on C; and "polars string buffers are
not zero-copy" was true of the accessor everyone used and **false of the
memory** — which is what section 8 turns on.

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
| **nanoarrow accessors** | use a library instead of hand-decoding the Arrow layout | **done — does not pay off** |
| **Arrow strings** | read polars' string bytes straight into the tree, no encoding | **done — this is the answer** |
| **polars Rust plugin** | make a tree a polars expression instead of a numba kernel | **done** |

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

**"polars string buffers are zero-copy, there is nothing to pack."**
Half right, and the half that was wrong sent three strands down a slower path.
`Series._get_buffers()` — what all of them used — *does* copy: it converts
polars' internal layout into a different one on every call, at 15–19 ns/row and
~44 MB per call on a 4-million-row column. But the memory itself **is**
reachable with no copy at all, through the Arrow C Data Interface
(`__arrow_c_stream__`), which is flat at ~1.2 µs however many rows there are
and adds no resident memory. Section 8. The accessor was the copy, not the
memory — so every "lazy" number in sections 4 and 5 was carrying a tax that
does not have to exist.

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

Inlining is **cheaper to compile cold and about 0.2 s dearer to start warm**.
The reason is worth knowing: numba's inlining runs before type inference, so in
`fused` mode the per-row kernel absorbs the walker *and* the tree's `path_fn`.
`path_fn`'s cached entry still exists and is still used by the `interpreted`
and `stepped` modes; in `fused` mode both bodies are compiled into the kernel,
which was never cacheable anyway because it captures a compiled function. So
nothing is lost that was being saved — but a warm `fused` start pays about
0.2 s more than it did. Typed features cost about 1.4× the cold compile and
1.5× the warm start, for the same reason plus more specialisations.

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

## 7. What I would do, in order

Nothing here needs a decision before you have had coffee. This is the order I
would take them in, cheapest and most certain first.

### 1. Take the inlining. It is ready, on a branch, and it is about 2×.

`walk_tree` becomes `@njit(inline="always")`; the tree's `path_fn` becomes
`@njit(cache=True, inline="always")`. Branch
`worktree-agent-a4e1be627b635397e`, commit `cdabaf1` — about 100 lines of real
change plus a new EXPERIMENTS.md §X. Not pushed, not merged; it is yours to
look at.

- **Tree: 1.6–1.9×** — measured there independently of me, three interleaved
  rounds, and it agrees with mine.
- **Branch/Loop: 22%** — 627 → 489 ns/row. The same rule applied to the
  control-flow walker.
- **Tables: no.** 2%, inside noise — a table's cost is the 150-cell scan, not
  the call. Left alone, correctly.
- **The string matcher: no.** Neutral; nothing array-shaped crosses its call.
- Output byte-identical on all five shapes; `build --verify`'s "no
  compilation after warm-up" property still holds cold and warm; and I ran the
  suite on the branch myself — **549 passed**.
- One cosmetic thing you would otherwise notice: the warning count in a test
  run goes from 17 to 58. They are all the same pre-existing
  `NumbaExperimentalFeatureWarning` about first-class function types, reported
  from more call sites now that the inlined bodies land in more callers. No new
  kind of warning appeared.

It has nothing to do with strings or types — it fell out of the typed work —
and it is the largest single number of the night.

### 2. Extend string matching past `exact` with the per-category mask.

No new dependency, no C, no Rust, no crash surface, and at 1.07 ns/row against
34 it is the fastest option for the columns real credit rules use — `sector`,
`product_code`, `employer_type`. It works by running the pattern over the
distinct values once per batch and having the kernel do one array lookup per
row. **Check the precondition first**: it only wins while the column arrives
already dictionary-encoded, which decider2 does today. There is a cardinality
threshold past which it collapses (a million distinct strings cost more to
encode than to match); keep the frame tier for those.

### 3. Decide on typed features as a correctness change, not a speed one.

It fixes a silent wrong answer on the column type doc 03 §1.2 mandates for
money, and turns `sector < 5` on a string column from a meaningless comparison
into a build error. It costs about a fifth of the walk and 1.4× the cold
compile. Both those prices are worth paying. Two things to go in with your eyes
open: the fix is **opt-in**, so existing trees keep the bug until they declare;
and a computed expression still cannot read an int- or bool-typed feature,
which is a loud error rather than a silent widening.

My read: yes, merge it — but the reason is the wrong answer and the build-time
rejections, not the benchmark in its write-up.

### 4. Do not adopt Rust or C for string matching.

Both work. Both cache correctly. Both are within noise of each other at 30–40 ns
per regex call. Neither is worth it:

- the entire prize is 20–40 ns/row, on a path where decider2's own marshalling
  already costs 350–1500 µs per `score()` call;
- C brings a backtracking engine, and a rule author writing `^(a+)+$` — which
  looks innocent — costs 211 µs/row with a safety limit and 42 ms/row without;
- Rust brings a wheel per platform and a failure mode that aborts the process
  rather than raising.

Keep both directories. They are documented, reproducible, and if a profile ever
shows a selective regex over a high-cardinality column, the answer is already
written — and on that day, pick Rust, because linear-time matching is the axis
that matters, not speed.

### 5. Then reconsider where the time actually goes.

Every engine measured tonight sits 150–1000× below decider2's own `score()`
floor. Marshalling a row in and reading an answer back is the whole cost on the
realtime path, and doc 05 §3.1b's fix for it is still unbuilt. That is where
the next real win is — not in the kernel.

### What I got wrong, collected in one place

- **"polars string buffers are zero-copy."** They are not, on polars 1.41.
  Asking for them materialises them every call, at 7–11 ns/row. Both string
  strands found this independently, and I verified it myself.
- **"the Rust regex crate is 25.6 ns/call."** ~40 ns on realistic string
  lengths; the fixed per-call overhead dominates short haystacks.
- **"C regex is too slow from a kernel" (EXPERIMENTS.md §T).** That was glibc,
  not C. PCRE2 with JIT is 2–7× faster and level with polars' own engine.

---

## 8. Strings straight out of polars, through Arrow — and this is the answer

You asked whether Arrow could let a string reach the tree without being turned
into a number first, and said you would pay more for the simpler pattern. It
turns out to be **cheaper**, not more expensive.

### The thing I had wrong, and it matters

Every strand above measured polars' string buffers through
`Series._get_buffers()` and concluded they are not zero-copy. That accessor
does copy. The memory does not have to be:

| | cost at 100k rows | at 1M | at 4M | memory added |
|---|---|---|---|---|
| `__arrow_c_stream__()` — the Arrow C Data Interface | 1.2 µs | 1.3 µs | 1.2 µs | **0 MB** |
| `_get_buffers()` — what everyone used | 1.5 ms | 19 ms | 78 ms | ~44 MB per call |

I checked this myself two ways that need no knowledge of the layout: the cost
is **flat** in rows for one and linear for the other, and five calls on a
76 MB column added **0.0 MB** of resident memory for one and **222 MB** for the
other. polars hands over its own buffers, with an owned reference — you can
drop the DataFrame and the bytes stay valid.

So the accessor was the copy. Every "lazy" number in sections 4 and 5 was
paying a tax that did not have to exist.

### What that makes possible

A tree node that reads the row's actual bytes and matches them in place — in
**pure numba**, no C, no Rust, no dictionary encoding, no pass over the column:

```python
elif k == STR:                       # a string test, at the node
    view = views_addr + 16 * row     # polars' own 16-byte view for this row
    n    = load_u32(view)            # length
    if n <= 12:                      # short string: the bytes are inline
        hit = cmp_inline(view + 4, n, pat_addr, pat_len, mode)
    else:                            # long: (buffer index, offset) into a data buffer
        hit = cmp_buffer(data_tab, view, n, pat_addr, pat_len, mode)
    pc = then_[pc] if hit else else_[pc]
```

`exact`, `starts_with`, `ends_with`, `contains` — all four, at the node. 38
tests pass; I ran them myself. They cover nulls, empty strings, multi-byte
UTF-8, sliced frames, all-null and zero-row columns, and multi-chunk frames.

### What it costs — cheaper, in most cases by a lot

1M rows, ns/row, boundary plus kernel:

| | today | STR node |
|---|---|---|
| low cardinality, `== "dog"` | 178–185 | **56–75** |
| high cardinality, `== "dog"` | 800–1100 (the encoding dominates) | **38–55** |
| `contains`, low cardinality | 66–155 (precomputed in polars) | 98–110 |
| **gated — 1% of rows reach the node** | 62–198 | **17–33** |

The gated row is your original argument, finally paid off: a tree
short-circuits, a column-wide pass cannot. 3–6× there.

A `STR` node does cost 25–40 ns more than a numeric node when it is reached.
End to end that is swamped by not encoding a million strings.

### The simplicity verdict, which is what you actually asked for

| | today | with the STR node |
|---|---|---|
| a string arrives as | an int32 code, from a `cast(pl.Categorical)` every batch, plus a category list | the bytes polars already holds |
| preprocessing pass | yes, every batch | none — one ~1 µs handshake per column |
| non-exact match | refused; author precomputes a bool column | four match types at the node |
| a pattern in a rule is | resolved to a code against *this batch's* category list | bytes in a table, an argument like any threshold |
| a maintainer must understand | codes, category lists, code stability across batches, the float64 slot the code rides in, the frame-tier workaround | 16 bytes per row: a length, then bytes or (buffer, offset) |
| nulls | `NaN` in a float slot | a validity bit; a null never matches |
| still float-encoded | yes | **no — a string stays a string** |

### What still has to stay, honestly

- **`regex` is not covered.** nopython numba has no regex engine. If a rule
  needs a real regex, the frame-tier workaround stays for that one case — and
  that is the only remaining argument for the C or Rust strands.
- **`Categorical`/`Enum` columns** arrive from upstream already as codes and
  keep the existing path. That is polars making a distinction, not the boundary
  inventing one, and no cast happens either way.
- **`isin` over thousands of values** becomes k byte-compares. Fine for a
  handful, not for thousands.
- **Case-insensitive and trim** are a few lines for ASCII; Unicode case folding
  needs a table and is not in reach without a dependency.

### One correctness trap worth knowing about

A polars Series can hold several **chunks**. Naive buffer access reads only the
first and silently gives wrong answers for the rest. The prototype walks chunk
by chunk and proves it (a 5+5 concatenation answers on all ten rows), and where
two columns of one frame have *different* chunk layouts it refuses loudly and
tells the caller to `rechunk()`. Whoever lands this must keep that property —
it is exactly the kind of silent wrong answer doc 03 §2.1 calls the worst
failure the design can have.

### How it fits the typed-features branch

That branch already reserved a raw-string slot, and it is the right idea with
the wrong plumbing: it carries `_get_buffers()`'s offsets and values — the
15–19 ns/row copy. Swapping that for the C Data Interface's view table removes
the copy, deletes its `_fill_spans` step, and gives `walk_tree` the `STR` kind.
The two pieces of work fit together directly.

Full detail: `experimentation/arrow-strings-in-tree/RESULTS.md`; `./run_all.sh`
reproduces it in about ten minutes.

---

## 9. The polars Rust plugin — more honest, not simpler, and it misses on your 90%

A real `pyo3-polars` plugin was built: 227 lines of Rust, 88 of Python, tree
passed as data, exposed as a polars expression.

### It works, and it is honest about types

The triple you care about, all passing:

| | plugin | decider2 today |
|---|---|---|
| `[2⁵³, 2⁵³+1] == 2⁵³` on Int64 | `[true, false]` | `[true, true]` |
| a Boolean column | stays `[1, 0, null]` | becomes float |
| a String column | matched as a string, no dictionary anywhere | int32 code |

It also *refuses* what it cannot do honestly — a Categorical column, a float
tested as an int, unsupported dtypes — each naming the node, the column and
both dtypes. That is the right behaviour and decider2 does not have it today.

### But the single-record case, which you say is 90% of traffic

µs per call, against your 60 µs spec:

| | 1 thread | 28 threads (default) |
|---|---|---|
| **the spec** | **60** | **60** |
| decider2 `score()`, same pipeline | 580–610 | 560–600 |
| plugin, best configuration | **236** | 694 (spread 270–1800) |

Two things to take from that.

**It beats decider2 by ~2.5× and still misses the spec by 4×.** Of that 236 µs,
the tree walk itself is **about 1 µs**. Everything else is engine machinery:
29 µs planner floor, 37 µs per plugin call, 19 µs to build a one-row frame.
This is the same wall decider2 already hit from the other side — the engine
does not matter at one record, the plumbing does.

**Default threading makes it three times worse.** polars dispatches a
multi-input expression's inputs to its thread pool even for a single row. So
single-record serving needs `POLARS_MAX_THREADS=1` — the exact opposite of the
setting that wins on batches. You would be running two configurations.

There is also a trap worth knowing: **the tree is deserialised on every
evaluation** — about 2 µs per node, so 1.34 ms for a 512-node tree. A packed
binary format cut that 11×, but it is a per-call cost that does not exist in
decider2 at all.

### On batches it wins clearly, from 100k rows up

ns/row, 4-step pipeline, answers checked against a pure-polars oracle:

| rows | decider2 fused (1 thread) | plugin, streaming, 28 threads |
|---|---|---|
| 10k | 464 | 242 |
| 100k | 252 | **37.8** |
| 1M | 253 | **17.2** |
| 10M | 931 | **11.0** |

At 1M rows decider2 takes 253 ms, outside its own 20–100 ms budget; the
streaming plugin takes 17 ms. That is real and it is 7–85× from 100k up.

### A correction I owe you

When you first raised this I said decider2's structural advantage is that N
steps compile into **one** kernel over one row loop, and that a polars plugin
would trade that away. **That is not true of trees.** decider2's own source
says so, in `compile/driver.py`:

> *"A real, reported scope cut, not full fusion. Each packed step becomes its
> OWN compiled segment — this does not fuse a tree's own matcher/path/output
> steps into ONE kernel call."*

Trees, tables, Branch and Loop are all "packed" steps, and none of them fuse.
Fusion applies to ordinary steps. So the 2.1× the plugin wins single-threaded
is mostly decider2 running five segments with intermediates — not Rust beating
numba. **The fusion argument I gave you does not apply to the case we were
discussing**, and the trade-off is better for the plugin than I said.

### What it costs to own — measured, and worse than the earlier Rust figures

| | earlier standalone Rust | this plugin |
|---|---|---|
| first build | 29 s | **5 min 38 s** |
| cold rebuild | 15 s | 4 min 39 s |
| one-line change | 0.9–1.5 s | 5.7–10.2 s |
| wheel | 888 KB | **5.71 MB** (`.so` 21.8 MB) |
| crates | 50 | **252** |

A plugin statically links its own copy of polars, which is where all of that
comes from. Your polars version is then chosen by `pyo3-polars`, not by you:
it pins `polars ^0.55`. Upgrading Python-side polars is safe; upgrading the
Rust side is a five-minute rebuild plus API churn.

### The one that would worry me in a credit path

**A column of a dtype the plugin was not compiled with aborts the Python
process with `SIGABRT` and no exception.** A `Decimal` column does it. The
panic happens inside polars' own FFI import, before the plugin's error
handling can see it. The mitigation is to build with `dtype-full`, but nothing
enforces that, and the failure is a process abort rather than a raised error.

Ordinary panics inside the plugin body *are* caught and become `ComputeError`.
This one is not.

### Verdict

**More honest, provably. Not simpler.** You would take on a Rust toolchain, a
five-minute build, 252 crates, a 5.7 MB wheel per platform, a polars version
you do not control, and a process-abort failure mode — to buy honest types
(which typed feature arrays already give you inside decider2 for ~20% of the
walk) and parallel batch speed.

And on the batch side there is a cheaper answer worth knowing: **plain
`pl.when/then` beats the plugin below about 32 leaves** with no Rust at all
(19.6 vs 39.5 ns/row at 8 leaves), and loses 11× at 512 leaves because it
evaluates every condition rather than short-circuiting. So the plugin's real
niche is *large* trees on *large* batches.

Your "depend on a maintained library" preference is real, but it applies to
the **bridge**, not the tree: `pyo3-polars` maintains the FFI; the ~300 lines
of tree logic, the wheel matrix and the rebuilds stay yours.

Full detail: `experimentation/polars-plugin-trees/RESULTS.md`, 248
measurements in `results.jsonl`.

---

## 10. nanoarrow — the library does not save you from hand-writing it

You asked to use an existing library rather than own the layout decoding, on
the grounds that if the spec changes you just pull the latest. That is the
right instinct in general. Here it does not pay off, for a specific and
checkable reason.

Three approaches were built and measured on identical data:

- **(A)** today's hand-decoded numba
- **(B)** vendored nanoarrow called per row through a C shim
- **(C)** nanoarrow `ValidateFull` once per batch, then per-row numba

### The finding that decides it

**nanoarrow does not validate string views at all.** I checked the vendored
source myself: `ArrowArrayViewValidateFull` has cases for `LIST_VIEW` and
`LARGE_LIST_VIEW` and **no case for `STRING_VIEW` or `BINARY_VIEW`**. It never
checks a single element's `(buffer_index, offset, length)`. So option (C)'s
whole premise — validate once, then the per-row path is provably safe — is
false. The prototype proves it: the variant that trusts `ValidateFull` passes
validation and then segfaults.

And the fast accessor is named `...Unsafe` because it is. Handed deliberately
corrupted arrays, each in its own subprocess:

| corruption | hand-decoded numba | nanoarrow (unsafe) |
|---|---|---|
| buffer index 1000 | error leaf | **SIGSEGV** (signal 11, confirmed) |
| offset 2³¹−1 | error leaf | **SIGSEGV** (signal 11, confirmed) |
| length 2³⁰, pattern absent | error leaf | **wrong answer** — matched bytes past the buffer |
| truncated buffer count | error leaf | **wrong answer** — nanoarrow computes −1 variadic buffers and accepts it |
| a `sizes` buffer that lies | wrong answer | wrong answer (nobody can catch this) |

nanoarrow's import check catches one of five. The hand-decoded kernel refuses
everything it can know about.

### So what does the library actually save you?

| | lines of Arrow-layout knowledge you own |
|---|---|
| hand-decoded numba | **42** (Python) |
| nanoarrow, unsafe accessor | **0** — and it crashes |
| nanoarrow + your own bounds checks | **37** (C) |

That is the whole answer. nanoarrow *decodes* the layout for you but does not
*check* it, so to get back to safety you write the bounds checks yourself — 37
lines of C instead of 42 lines of Python, plus the project's first compiled
artefact. **The library moves where you hand-write it, and makes it C.**

### Where nanoarrow genuinely wins, in fairness

If polars ever exported plain `u` (utf8) or `U` (large_utf8) instead of `vu`,
**nanoarrow handles it with zero code changes**; the hand-decoded version
refuses loudly (`expected 'vu'`) and would need 10–15 lines to support it. On a
schema that lies about its own type, though, nanoarrow segfaults and the
hand-decoded version refuses.

And "just pull the latest" is mechanically true — nanoarrow ships about two
releases a year with a stable C API — but the Arrow format is *additive*.
Utf8View has not changed since Arrow 15. A "spec change" means a *new* format,
which the hand-decoded version refuses by name, and which nanoarrow only helps
with once it supports it **and** the accessor is safe — which for string views
it is not.

### Speed at one record, which is your 90% case

µs per call, against the 60 µs spec:

| | as written | with the Python glue leaned out |
|---|---|---|
| hand-decoded numba | 42.6–45.1 | **21.0** |
| nanoarrow per row | 19.3–20.6 | 9.3 |
| validate-full + numba | 33.3–35.9 | 17.2 |

nanoarrow is genuinely faster — but look at *what* is being timed. In every
approach the dominant cost is the **import handshake and Python glue**, not
decoding: 26 µs of the hand-decoded path is `export`, of which ~8 µs is a
single `np.ctypeslib.as_array` call. Building the pattern table and the string
tables per call is another 7 µs of Python that should be hoisted. polars' own
`__arrow_c_stream__` is 0.5 µs.

**Leaning that glue takes 45 µs to 21 µs with no C at all** — most of
nanoarrow's advantage, none of its cost. At 1M rows the hand-decoded kernel is
actually 6–10% *faster*, because nanoarrow's per-row call cannot inline.

### Recommendation

**Keep the hand-decoded numba kernel, lean its glue, do not adopt nanoarrow,
and drop the validate-once idea entirely** — it rests on a check that does not
exist. Revisit only if the project ships a compiled artefact for some other
reason *and* nanoarrow gains string-view bounds validation. There is no
upstream issue tracking it.

### Two bugs found on the way

- **In our own prototype:** `pl.Series("s", [None])` infers dtype `Null`, not
  `String`, and `arrowc.export` dies on it with an `IndexError`. A real latent
  bug worth fixing.
- **In nanoarrow 0.9.0:** it accepts a string-view array declaring 2 buffers,
  computing −1 variadic buffers. Worth reporting upstream.

Full detail: `experimentation/nanoarrow-accessors/RESULTS.md`; `./run_all.sh`
reproduces in about six minutes.

---

*Still out: the pure-numba byte-matching strand, asked to wrap up with partial
results.*
