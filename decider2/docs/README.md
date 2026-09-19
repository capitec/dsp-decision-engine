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

Ten experiments are defined (E0–E9, doc 06) and **five have been run**: E0, E5,
E7, E8, E9. Five settled a structural question by measurement; **two returned
negative results that changed the design** rather than confirming it — fusion
turned out to be non-monotone, falsifying the stated reason for an authoring
recommendation, and a null-handling performance claim written into doc 01 proved
backwards. Both corrections are recorded in place rather than quietly fixed.

Still to run: **E1** (the polars↔numba boundary, "the one to build first"), E2
(the graph model), E3 (the equivalence ladder), E4 (the reviewable artefact —
people-blocked, and the top risk), E6 (param ergonomics), **E10** (configuration
lifecycle, doc 08 §9) and **E11** (`param()` in the signature, doc 06).

The highest remaining risk is not technical: it is whether a credit-risk reviewer
can actually verify a rule from the generated view (doc 04 §6). That one is
people-blocked and should start before implementation.

## Read in this order

| Doc | What it covers |
|---|---|
| [01-motivation-and-evidence.md](01-motivation-and-evidence.md) | Why a new engine. Every measurement and observed failure mode that drove a decision. |
| [02-architecture.md](02-architecture.md) | The tier model, execution modes, seams, package layout. |
| [03-authoring-api.md](03-authoring-api.md) | The developer-facing interfaces. How people actually write logic. |
| [04-observability-and-governance.md](04-observability-and-governance.md) | Who can change what, how a decision is explained, what survives an audit. Personas, the params/structure permission boundary, lineage, taps, trace, the audit record. |
| [05-boundary-and-compilation.md](05-boundary-and-compilation.md) | **Implementation spec for `compile/` — the first thing to build.** Extraction, nulls, calling convention, codegen, variants, fallback, fusion grouping, acceptance criteria. |
| [06-open-questions-and-experiments.md](06-open-questions-and-experiments.md) | What's still undecided, what's been settled and by which experiment. |
| [07-project-structure.md](07-project-structure.md) | How a project *using* the framework is laid out — modules, pipelines, config, defaults, the Python/JSON duality. |
| [08-configuration-and-lifecycle.md](08-configuration-and-lifecycle.md) | What config may change and what it may not; the three change classes; bounded module interiors; the compilation lifecycle and staged swap; where config comes from (and why that is not the framework's business). |

A standing review of this doc set, with findings ranked by cost of late
discovery, is in [REVIEW.md](REVIEW.md). Items acted on are marked there.

**[EXPERIMENTS.md](EXPERIMENTS.md) holds measured results** from seven experiments
run against real numba on this project's own environment. Five refuted or
partially refuted a documented claim; three changed a design decision. Every
harness is checked in under `experimentation/` and is runnable — unlike the
figures in doc 01, which existed only as tables and could not be re-run on a
different workload shape. Where a claim in docs 01–08 has been measured, an
inline note points here.

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
applied over polars columns; anything numba can't compile degrades to plain
Python per-node rather than failing. Tunable values live in one validated
per-invocation `params` bundle, separate from graph structure, which makes the
performance boundary and the governance boundary the same boundary. Four
execution modes share one definition, from fully-fused production to a
single-record step-through debugger, and their agreement is automatically tested.
