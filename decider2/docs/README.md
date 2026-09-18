# decider2 — design documentation

A numba-first decision engine for credit-granting logic. Polars owns set-shaped
frame operations (joins, aggregations); compiled scalar Python owns per-record
decision logic.

`decider2` is a working name. If the design holds up it replaces `decider`.

## Status

Design phase, doc set complete. **No code yet — deliberately.** These documents
are the specification; implementation follows once the interfaces are agreed.

Nine experiments were run before writing any framework code (doc 06). Five
settled a structural question by measurement; **two returned negative results
that changed the design** rather than confirming it — fusion turned out to be
non-monotone, falsifying the stated reason for an authoring recommendation, and
a null-handling performance claim written into doc 01 proved backwards. Both
corrections are recorded in place rather than quietly fixed.

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

## Conventions used in these docs

- **Settled** — decided, with evidence recorded in doc 01.
- **Proposed** — a concrete recommendation awaiting review. Most interface
  details in doc 03 are proposals; they exist in written form precisely so they
  can be argued with.
- **Open** — genuinely undecided, tracked in doc 06.

### Citing the current implementation

Where these docs describe how `decider` behaves today, the claim is cited as
`decider/<path>:<line>` so a reader can check it rather than take the doc's word.

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
