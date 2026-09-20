# Evaluation brief — implement a spec with decider2

You are implementing **one** example project spec using `decider2`, a
numba-first decision engine for credit and fraud logic.

**This is an evaluation of the framework, not of you.** Where you struggle, the
framework is probably wrong. Say so plainly — that is the most valuable thing
you produce.

## Ground rules

1. **Do not open `example_projects/examples/`.** It contains mock
   implementations written before the framework existed, against an imagined
   API. Reading them will teach you the wrong thing. If you catch yourself
   wanting to, write down what you wanted from it instead.
2. Implement the **whole** spec, not a slice. If you run out of room, finish
   what you can and list precisely what you left and why.
3. It must **run as an HTTP endpoint** and answer a real record.
4. Write tests. Use `decider2.testing.assert_equivalent` on at least one
   pipeline — it runs all three execution modes and asserts they agree.

## The library, in one page

```python
from decider2 import flow, module, param, step, missing_as

# A step is a plain function. Its NAME is the output. Its PARAMETERS are the
# inputs, matched BY NAME to other steps' outputs or to frame columns.
# The DOCSTRING is the description a reviewer reads.
def disposable_income(net_income: float, expenses: float) -> float:
    """Income remaining after committed expenses."""
    return net_income - expenses

# param() makes a value tunable WITHOUT moving it out of the function.
# It takes exactly pydantic's Field arguments.
def cap_by_income_band(
    term_cap: float,
    min_net_salary: float,
    cap: float = param(48.0, ge=6, le=60),
    floor: float = param(5000.0, ge=0),
) -> float:
    """Cap term below the income floor.

    Implements: Credit Policy §7.4.2
    """
    return min(term_cap, cap) if min_net_salary < floor else term_cap

# flow() builds a pipeline. `|` also works once you have a module on the left.
pipeline = flow(disposable_income, cap_by_income_band)

pipeline.apply(frame)                     # batch, polars in/out
pipeline.score({"net_income": 9200.0})    # one record, dict in/dict out
```

- `@step(output="term_cap")` only when the output name differs from the
  function name. `@step(nogil=True)` to release the GIL (off by default).
- `module(a, b, c, name="x", params=Model)` when several steps share knobs.
- `.emit("x")` to keep an intermediate in the output frame; `.emit("x@*")` for
  every version of a rewritten name; `.drop("x")` to remove a column.
- `.relabel(reads={...}, writes={...})` to adapt a module to different names.
- Money is **scaled int64 cents**, never float. Use `round_half_up`.
- A `str` input enters the kernel as a dictionary code, so a string literal you
  compare against must be declared as a `param("literal")`, not written inline.

**Read `docs/03-authoring-api.md` first.** It is the authoring surface. Consult
`docs/07-project-structure.md` for layout and `docs/08-configuration-and-lifecycle.md`
for params/config. Do not try to read all the docs.

## Serving it

```bash
../../.venv/bin/python -m decider2 serve <your_pipeline.py> --port <port>
```

Endpoints: `/ping`, `/invocations` (POST a record), `/params`, `/params/schema`,
`/params/preview`, `/rollback`, `/health`.

## What to report back

1. **What you built**, and what you left out.
2. **Where the framework fought you.** Be specific and quote the doc sentence
   that misled you, if one did. This is the point of the exercise.
3. **What you had to guess** because no doc said.
4. **What you wanted and could not express at all.**
5. The exact `curl` that proves your endpoint answers, and its output.
