# decider2 — design documentation

A numba-first decision engine for credit-granting logic. Polars owns set-shaped
frame operations (joins, aggregations); compiled scalar Python owns per-record
decision logic.

`decider2` is a working name. If the design holds up it replaces `decider`.

## Status

Design phase, doc set complete. **No code yet — deliberately.** These documents
are the specification; implementation follows once the interfaces are agreed.

`decider2` is a **ground-up build.** No part of the current implementation is
carried forward, and where these documents describe `decider`'s behaviour they do
so as evidence about what a design compels, not as a base to extend. Doc 01 §5 is
the concentrated version of that evidence.

**Twenty experiments have now been run against real numba** (EXPERIMENTS.md).
Most refuted or partially refuted a documented claim, and several changed a design
decision rather than a number: fusion and `prange` became authored rather than
inferred, per-node fallback turned out to be impossible inside a fused driver, the
output convention moved to dtype-grouped 2D arrays, chunking became mandatory, the
compile lifecycle moved from a thread to a subprocess, and the single-record path
turned out to cost a whole millisecond (4.9% of a 20 ms budget) as its own
conventions most naturally imply implementing it — fixable to 0.73% by marshalling
and reading the record a row at a time instead of a field at a time. N2 is a full
reversal of doc 02 §3.5's own example at width: calling `score()` with 400 literal
keyword arguments, as specified, costs 5.95% of a 20 ms budget on the calling
convention alone — more than everything else in the framework combined — because
CPython's keyword-argument binding scales close to quadratically with parameter
count; a dict costs 20× less for the same data. `score()` now takes a dict. The
newest one (N3) is a confirmation rather than a refutation: params validation on
the request path (doc 02 §4's "params may arrive per invocation") costs at most
1.28% of a 20 ms budget, even at 50 module instances — affordable as written, so
the question of whether to restrict realtime payload params now rests on
governance (doc 04 §2.1) alone, not performance. **N4, the last item in the
single-record set, measured the tail** (p50–max, not just medians) for the first
time: a config-generation swap under continuous traffic is confirmed cheap for
serving (worst call across 30 swaps: 1.37% of budget), GC on/off/frozen makes no
measurable tail difference, and the one finding that changes a recommendation is
concurrency — serving kernels want `nogil=True` (authored per step, doc 00 §2c), because
the identical kernel compiled `nogil=False` hits a GIL convoy effect that blows
the tail to **12.7× the entire 20 ms budget** at 16 concurrent threads, even
though total throughput is unaffected either way. The twentieth (M) closes the
one disagreement between two prior experiments: §K and §L attributed the same
stale-constant hazard to different caches, and both were right — CPython's
`__pycache__/*.pyc` and numba's own on-disk cache are each independently
sufficient to serve a stale value, depending on whether the edited constant
collides with another one in `co_consts`; content-addressed naming was always
the fix either way, now confirmed against the layer actually responsible. Every
correction is recorded in place rather than quietly fixed.

Still to run: **E1** (the polars↔numba boundary, "the one to build first"), E2
(the graph model), E3 (the equivalence ladder), E4 (the reviewable artefact —
people-blocked, and the top risk), and E6 (param ergonomics). N1–N4, the whole
single-record/tail/concurrency set doc 01 §6.1 called highest-value, are all
now done.

The highest remaining risk is not technical: it is whether a credit-risk reviewer
can actually verify a rule from the generated view (doc 04 §6). That one is
people-blocked and should start before implementation.

## Read in this order

> **Start with [00-BUILD.md](00-BUILD.md).** These documents were written over
> several rounds and twenty experiments later refuted a number of their claims.
> The corrections are recorded in place, so reading linearly you will meet a
> confident wrong statement before its retraction. 00-BUILD lists every superseded
> claim in one table, says what is settled, and gives the build order with the
> open question that blocks each layer.

| Doc | What it covers |
|---|---|
| [00-BUILD.md](00-BUILD.md) | **Read first.** Superseded claims, what is settled, build order, and which open questions block which layer. |
| [01-motivation-and-evidence.md](01-motivation-and-evidence.md) | Why a new engine. Every measurement and observed failure mode that drove a decision. |
| [02-architecture.md](02-architecture.md) | The tier model, execution modes, seams, package layout. |
| [03-authoring-api.md](03-authoring-api.md) | The developer-facing interfaces. How people actually write logic. |
| [04-observability-and-governance.md](04-observability-and-governance.md) | Who can change what, how a decision is explained, what survives an audit. Personas, the params/structure permission boundary, lineage, emitted values, trace, the audit record. |
| [05-boundary-and-compilation.md](05-boundary-and-compilation.md) | **Implementation spec for `compile/` — the first thing to build.** Extraction, nulls, calling convention, codegen, variants, fallback, fusion grouping, acceptance criteria. |
| [06-open-questions-and-experiments.md](06-open-questions-and-experiments.md) | What's still undecided, what's been settled and by which experiment. |
| [07-project-structure.md](07-project-structure.md) | How a project *using* the framework is laid out — modules, pipelines, config, defaults, the Python/JSON duality. |
| [08-configuration-and-lifecycle.md](08-configuration-and-lifecycle.md) | What config may change and what it may not; the three change classes; bounded module interiors; the compilation lifecycle and staged swap; where config comes from (and why that is not the framework's business). |

A standing review of this doc set, with findings ranked by cost of late
discovery, is in [REVIEW.md](REVIEW.md). Items acted on are marked there.

**[EXPERIMENTS.md](EXPERIMENTS.md) holds measured results** from twenty
experiments run against real numba on this project's own environment. Most
refuted or partially refuted a documented claim, and several changed a design
decision rather than a number — fusion, `prange`, per-node fallback, the output
convention, the compile lifecycle and the single-record marshal/readback shape
all moved. Every harness is checked in under
`experimentation/` and is runnable, unlike the figures in doc 01, which existed
only as tables and could not be re-run on a different workload shape. Where a
claim in docs 01–08 has been measured, an inline note points here.

**[COLD-READ.md](../example_projects/COLD-READ.md)** reports a comprehension study:
sixteen readers (eight Sonnet, eight Haiku) given identical prompts, six reading one
mock project each with no framework docs and no spec, then checked against the spec.
Where a reader confidently concluded something the spec contradicts, the authoring
surface failed to communicate — those divergences, not opinions, are its findings.
It agrees with `example_projects/examples/FINDINGS.md` on four points by a method
sharing none of its assumptions; where they disagree it says which to trust.

**Read the performance findings against doc 01 §6.1.** The single-record path is
primary, with a 20–100 ms budget, and the compiled path runs at ~1 µs — so fusion,
output conventions and chunking are *batch* concerns with four orders of magnitude
of headroom at N=1, and should be chosen for maintainability unless a request-path
measurement says otherwise.

## Conventions used in these docs

- **Settled** — decided, with evidence recorded in doc 01.
- **Proposed** — a concrete recommendation awaiting review. Most interface
  details in doc 03 are proposals; they exist in written form precisely so they
  can be argued with.
- **Open** — genuinely undecided, tracked in doc 06.

### Citing the current implementation

Where these docs describe how `decider` behaves today, the claim is cited as
`decider/<path>:<line>` so a reader can check it rather than take the doc's word.

> **Which tree.** Some citations resolve only against an unpushed local branch —
> notably everything referencing `decider/modules/record.py` (`_build_jit_driver`,
> the `record` kind), `decider/modules/primitives/branching.py`
> (`how="diagonal_relaxed"`), `decider/plan.py` (`ExecutionPlan.execute(audit=True)`),
> `name_override`, and `experimentation/steptree_poc/jit_codegen.py`. On
> `origin/main` those paths do not exist and nothing under `decider/` imports
> numba. Line numbers throughout have also drifted by a few lines against either
> tree. **Re-run the citation pass against a pushed commit before treating doc 01
> as a checkable evidence base**, and pin that commit here.

Where a *consequence* is quantified — how many workarounds a framework behaviour
forced, for instance — the counts come from a large internal project built on
`decider`. That project is deliberately **not described**: no domain logic, field
names, thresholds, rule identifiers or file paths from it appear here. Only the
framework-ergonomics evidence does, and all worked examples in these documents
are illustrative and fictional.

Doc 01 §5 is the concentrated comparison: each `decider` behaviour, where it
lives in code, and what `decider2` does instead.

## The one-paragraph summary

Business logic is written as small, pure Python functions over scalars. Functions
are wired into a graph by matching parameter names to other functions' outputs.
The graph is plain validated data, not generated classes — so it can be rendered,
diffed and safely edited by tools. Steps compile to a single fused numba kernel
applied over polars columns; a step numba can't compile splits the kernel around
it rather than failing, since a nopython driver cannot call back into Python
(EXPERIMENTS.md §B). Tunable values live in one validated
per-invocation `params` bundle, separate from graph structure, which makes the
performance boundary and the governance boundary the same boundary. Four
execution modes share one definition, from fully-fused production to a
single-record step-through debugger, and their agreement is automatically tested.
Fusion and parallelism are authored, not inferred — measurement showed no constant
and no warmup heuristic can decide either one.
