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
  config/
    shared.json                 # SharedParams values
    term_loan/
      production.json           # explicit and complete
      staging.json
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

The rule, which is cheaply lintable: **every `def` in a pipeline file must be
referenced by a `module(...)` call in that file.** Steps may not float.

---

## 4. Pipelines as Python or as config

Both are first-class and produce the same object, because structure is data
(doc 02 §2). Python builds it with `module(...)` and `|`; JSON builds it with
`{"use": <id>, ...}` resolved through the discriminated union (doc 02 §2.1).

Combinators serialise too, since they are structure rather than logic:

```json
{"type": "branch",
 "condition": "credit:is_private_sector",
 "arms": [{"use": "credit:cap_private"}, {"use": "credit:cap_public"}],
 "modifies": ["term_cap"]}
```

**One honest asymmetry:**

| | compose modules | set params | define new logic |
|---|---|---|---|
| `pipeline.py` | yes | yes | yes — inline modules |
| `pipeline.json` | yes | yes | **no** |

JSON cannot define logic; code is code. The useful consequence is that **an
inline module is not addressable from JSON until it is registered** — which means
importable, which means it lives in `modules/`.

So "move it into `modules/` and register it" becomes the natural price of making
something reusable from config. Proportionate, and it keeps the JSON surface
referencing only properly packaged things. A small pipeline can stay Python
forever and never pay it.

---

## 5. Running, exporting, and the audit artefact

```
decider run    pipelines/term_loan.py
decider run    config/term_loan/production.json
decider export pipelines/term_loan.py -o term_loan.json   # fails if any module is inline
decider build  term_loan                                  # AOT compile, doc 02 §3.4
```

The workflow this supports: **prototype in Python, export to JSON for
deployment.**

And `export` output *is* the explicit, complete, fully-resolved config that §2
argues production needs for auditability — one artefact serving both purposes
rather than two that can drift. The export also mechanically proves every part of
the pipeline is registered and addressable, which is a stronger guarantee than a
review.

---

## 6. Lint rules worth enforcing

Cheap to implement, each preventing a failure mode observed in the current
project (doc 01 §5):

| rule | prevents |
|---|---|
| every `def` in `pipelines/` is referenced by a `module(...)` in the same file | logic hiding in composition |
| every module directory has a corresponding test | one test file for 66 modules |
| no bare numeric literal in a step body where a param exists with that value | a parameter existing in two places with no link (§5.3 — `max_term: 84` in JSON *and* hardcoded in two modules) |
| `grep -r "@breaks_lineage"` reviewed in CI | lineage gaps accumulating unnoticed |
| production config is complete — no reliance on code defaults | audit requiring cross-reference to a commit |
