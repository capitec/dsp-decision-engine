# 07 — Project structure

How a project that *uses* `decider2` is laid out. (The framework's own layout is
doc 02 §6.)

This applies identically to the shared credit-module library and to a client
extension project — a client extension is just this structure importing shared
modules alongside its own.

---

## 1. The layout

```
myproject/
  modules/                      # unit of reuse AND audit
    affordability/
      steps.py                  # the pure functions
      params.py                 # pydantic models with schema defaults
      __init__.py               # the module(...) assembly
    policy_rules/
      income_cap.py                   # a one-file module is fine
      sector_cap.py
    scorecards/
  pipelines/
    term_loan.py                # composition
    access_facility.py
  vocabulary.py                 # project name map, if the project's values differ
                                # from a shared library's (doc 03 §5.2)
  contracts/                    # frozen module interfaces, for published modules
    affordability.json
  config/                       # the DEFAULT layout the CLI assumes — one option,
    term_loan/                  # not an interface. Any loader returning a dict
      production.json           # works just as well (doc 08 §6).
      staging.json              #   params: explicit and complete, incl. shared
      rules/
        policy_rules.json       #   interior: the body of one ruleset
  schemas/
    term_loan_input.json        # declared input-frame schema — `build` needs it
  tests/
    modules/                    # rule-level intent assertions
    pipelines/                  # end-to-end, golden traces
```

Four principles hold it together:

1. **A module directory is the unit of reuse and audit.** Steps, params and
   assembly live together, so reviewing one rule means opening one place.
2. **Pipeline files are composition.** They may *declare* modules (§3) but must
   not contain logic that isn't inside one.
3. **Config mirrors pipelines, not modules** — params are namespaced by module
   *instance* name within a pipeline (doc 03 §4.1), so a module reused in two
   pipelines correctly has two config entries.
4. **Tests mirror modules**, which makes "module with no test" a detectable gap.
   Worth having, given the reference point: one test file for 66 modules, and
   `testpaths` configured so it didn't even run (doc 01 §5.6).

---

## 2. Where default parameter values live

Both in code and in config, with different jobs. This split exists for an
**audit** reason, not a convenience one.

| | holds | job |
|---|---|---|
| `params.py` — `Field(36.0, ge=6, le=60)` | the **schema** default | documents intent, guarantees a valid runnable state, is what tests use |
| `config/<pipeline>/<env>.json` | the **deployed** values | what actually ran, per environment |

**Production config should be explicit and complete** — every parameter stated,
none inherited from a code default. Otherwise answering "what were the parameters
on 3 March?" means cross-referencing a config file against a code version at that
commit, which is exactly the kind of archaeology the design is meant to remove.

So: defaults in code for development ergonomics, resolved config as data for
auditability, and a tool that materialises one from the other (§5).

This also gives the personas a clean split (doc 01 §6): a business user edits
`config/term_loan/production.json`, bounded by the validators declared in
`params.py`, and cannot reach structure.

---

## 3. Inline modules — for when a directory is overkill

A three-step pipeline shouldn't need `modules/foo/{steps,params,__init__}.py`.
A pipeline file may declare a module inline:

```python
# pipelines/small_check.py
from decider2 import module, pipeline
from decider2.frame import Join

def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses

def passes(disposable_income: float, params) -> float:
    return 1.0 if disposable_income > params.floor else 0.0

Checks = module(disposable_income, passes, name="checks", params=CheckParams)

pipeline = Join(bureau, on="client_id") | Checks
```

Nothing is special-cased. `Checks` still has a name, a params namespace
(`{"checks": {...}}`) and an audit identity — it just has no directory. The
engine cannot tell the difference.

**The growth path is a pure move, with no semantic change at any step:**

```
inline in pipelines/x.py  →  modules/checks.py  →  modules/checks/
```

The rule, which is cheaply lintable: **every `def` in a pipeline file must appear
in the pipeline expression or in a `module(...)` call in that file.** Steps may
not float.

A bare function in the pipeline expression is itself a module (doc 03 §5.3), so
the smallest useful thing is one function and one line — not a directory.

---

## 4. Pipelines as Python or as config

**The pipeline skeleton is Python.** Which modules exist and how they wire is
composed with `module(...)` and `|`, reviewed in git, and changed by an engineer.
Config fills in values and the interiors of data-shaped modules — doc 08 §2.

| | compose modules | set params | edit rules in a `ruleset` | define new logic |
|---|---|---|---|---|
| `pipelines/term_loan.py` | yes | yes | yes | yes — inline modules |
| params document | **no** | yes | no | no |
| interior document | **no** | no | yes | no |

This is narrower than an earlier draft of this section, which granted JSON the
power to compose modules and declare `Branch` nodes. That was withdrawn for three
reasons, all in doc 08 §2: a composed graph arriving at runtime makes static
lineage a promise about something nobody reviewed; it puts a graph nobody reviewed
behind the audit hash; and it gains nothing a UI actually needs, because the
UI-editable surfaces are params, tables and rules — all of which are *interiors*
of a declared module.

**The option is deliberately left open.** The graph is a pydantic instance either
way, so widening this is an *admission policy* rather than a redesign
(doc 08 §7) — one call site, no new machinery, because interiors already require
everything composition would. And if a pipeline-builder UI is wanted, the better
answer is to have it **emit Python** and commit it: the graph stays in git, gets
reviewed, gets a commit hash, and a generated pipeline is indistinguishable from
a hand-written one.

The useful consequence of the module/interior split is unchanged: **an inline
module has no interior document until it is registered** — which means importable,
which means it lives in `modules/`. "Move it into `modules/` and register it"
stays the natural price of making something configurable. A small pipeline can
stay pure Python forever and never pay it.

---

## 5. Running, exporting, and the audit artefact

```
decider run     pipelines/term_loan.py --params config/term_loan/production.json
decider export  pipelines/term_loan.py --params -o config/term_loan/production.json
decider export  pipelines/term_loan.py --interiors -o config/term_loan/rules/
decider build   term_loan --schema schemas/term_loan_input.json   # AOT, doc 02 §3.4
decider config  fill config/term_loan/production.json --from term_loan
```

Three things to note, all consequences of doc 08:

- **`export` is split by document kind, not merged.** `--params` materialises
  every parameter at its current resolved value; `--interiors` writes each
  data-shaped module's body. They version independently: a params retune must not
  change the structure fingerprint, and it would if they shared a file.
- **`build` takes an input schema.** Column dtypes and nullability determine the
  record dtype, the numba signature and the `float64`-vs-`Optional` choice per
  step, so the pipeline alone is not enough to compile (doc 05 §8, O11).
- **`config fill`** writes newly added params at their schema default and leaves
  existing values untouched, so a library upgrade that adds a parameter lands as
  a reviewable diff rather than as a simultaneous hard failure in every consuming
  project. **It also seeds a new rule's first dated entry from the rule's own
  interior defaults**, which is otherwise the single largest source of hand-copied
  duplication in the corpus: `rule-sheet-MS-0208` restates ~13 values — including
  the entire `approval` block — across two files, with nothing checking that they
  agree. A first entry is derivable from the interior; only *subsequent* dated
  entries are genuine policy input. Seeding it is worth more than every other
  redundancy in this document combined, because it is the one that silently goes
  wrong rather than merely costing keystrokes.

The workflow this supports: **prototype in Python, export the documents for
deployment, keep the skeleton in git.**

`export --params` output is the explicit, complete, fully-resolved params document
that §2 argues production needs for auditability. Pipeline identity is *not* its
hash — that is params identity. What identifies the thing that ran is the
structure fingerprint plus the compiled artefact id (doc 08 §8), which is why the
two are recorded separately rather than merged into one file.

---

## 6. Lint rules worth enforcing

Cheap to implement, each preventing a failure mode observed in the current
project (doc 01 §5):

| rule | prevents |
|---|---|
| every `def` in `pipelines/` appears in the pipeline expression or in a `module(...)` in the same file | logic hiding in composition (doc 03 §5.3) |
| a module does not mix `param()` defaults and an explicit params model | two declarations of the same knob (doc 03 §4.4) |
| every module directory has a corresponding test | one test file for 66 modules |
| no bare numeric literal in a step body where a param exists with that value | a parameter existing in two places with no link (§5.3 — `max_term: 84` in JSON *and* hardcoded in two modules) |
| `grep -r "@breaks_lineage"` reviewed in CI | lineage gaps accumulating unnoticed |
| production params document is complete — no reliance on code defaults | audit requiring cross-reference to a commit |
| no I/O import (`json`, `os`, `pathlib`, `socket`, HTTP) under `binding/` or `params/` | the framework re-acquiring config sourcing (doc 08 §6) |
| `origin=` is never a string literal at a serving or pipeline entry point | provenance degrading to `origin="prod"` |
| `from_config(pipeline.to_dict()) == pipeline` round-trips in CI | the graph data model drifting into something config cannot express (doc 08 §7) |
| no node id generated with `uuid` | path codes incomparable across versions, and non-deterministic codegen (doc 08 §3.1) |
| no `{module_name, function_name}` pointer in any config document | code resolved by `getattr` with no declared interface or schema (doc 08 §1.1) |
| no expression string in any config document | a second, unspecified way to write arithmetic (doc 08 §3.2) |
| no bare `round()` in a step body — use `round_half_up` | njit and CPython round differently; one cent per disagreement (doc 03 §1.2) |
| no `Decimal` in a step signature or an input schema | raises a Rust panic at the boundary that `except Exception` misses (doc 05 §1.5) |
| no `int64` accumulator over a money column | wraps at 2,667 rows on realistic loan sizes (doc 03 §1.2) |
| every published module has a frozen `contract=` file, checked in CI | a library interface changing without its consumers knowing (doc 03 §5.1) |
| no identity-passthrough step (`return <param>`) | the workaround for a missing relabel — 79 of them in one project (doc 01 §5.1) |
| no `@step(output=...)` where the output equals the function name | 92 sites restating the default (doc 03 §1) |
| no `description=` on `@step` — the docstring is the description | 107 sites populating the governance artefact from the one field the guidance tells you to omit (doc 03 §1) |
| no `name=` on a single-step `module(...)` where it equals the function name | the same name written twice, 100 of 204 sites |
| no `contract=` string that equals `contracts/{module_name}.json` | 26 of 44 sites spelling out the derivable default (doc 03 §5.1) |
| no unused `from decider2 import step` | 35 files importing a decorator they no longer use |
| a rule's first dated params entry agrees with its interior defaults | ~13 values hand-copied across two files with nothing checking them (§5, `config fill`) |
| every rule function's name contains its policy rule id | the spec↔code join that made verification mechanical rather than interpretive — "the single biggest comprehension aid in the whole project" (COLD-READ §4.2) |
| a `ruleset` never reads or writes a name outside its declared bound | a config edit widening its own blast radius (doc 08 §2) |
