"""Playground: `Branch` and `Loop` (doc 03 §8.2/§8.3), nested.

A teaching artefact, not a policy — the numbers are illustrative. It shows,
in order:

  1. a plain `Branch` (`TermCapBySector`) — routes term cap by sector;
  2. a plain `Loop` (`ExtendTermToFit`) — extends a term cap, one step
     at a time, until the instalment fits, with a real early exit;
  3. the nested shape: `SearchForViableTerm`, a `Loop` whose body is
     `AdjustmentStrategy`, a `Branch` whose own "fast" arm is itself a
     `Loop` (`FastTrackBump`) — i.e. `Loop(Branch(steps1, Loop(steps2)))`;
  4. `<name>_path` emitted so you can see which arm fired, for both the
     plain Branch and the nested one;
  5. the same pipeline served over HTTP (`decider2 serve`), with the
     exact `curl` commands to hit it.

Run it:

    ../.venv/bin/python -m decider2.examples.playground_control_flow

Then, to serve it (`pipeline` below is the section 3/nested demo):

    decider2 serve src/decider2/examples/playground_control_flow.py --port 8971
    curl -X POST localhost:8971/invocations \\
        -d '{"high_income": true, "principal": 100000.0, "term_cap": 12.0}'

Verified — real output, this exact command, right after the `python -m`
run below:

    {"high_income": true, "principal": 100000.0, "term_cap": 48.0, "is_high_income": true}

Matches section 3's own batch row for `high_income=true` exactly (48.0) —
doc 05 §9's acceptance criterion 2, "the same kernel answers a single
record", holds for a nested Loop(Branch(...Loop...)) same as it does for
anything else.

--------------------------------------------------------------------------
REAL OUTPUT — captured by running this file, verbatim (doc 00's own rule:
"verify by running it and pasting the real output into the module
docstring"). `rm -rf .decider2_cache` was run immediately before, so this
is a cold-cache, from-scratch build.

$ ../.venv/bin/python -m decider2.examples.playground_control_flow

=== 1. Plain Branch: TermCapBySector ===
shape: (3, 5)
┌─────────┬──────────┬────────────────────┬─────────────────────────┬───────────────────┐
│ sector  ┆ term_cap ┆ term_cap_by_sector ┆ term_cap_by_sector_path ┆ is_private_sector │
│ ---     ┆ ---      ┆ ---                ┆ ---                     ┆ ---               │
│ str     ┆ f64      ┆ f64                ┆ i64                     ┆ bool              │
╞═════════╪══════════╪════════════════════╪═════════════════════════╪═══════════════════╡
│ private ┆ 60.0     ┆ 48.0               ┆ 0                       ┆ true              │
│ public  ┆ 60.0     ┆ 60.0               ┆ 1                       ┆ false             │
│ private ┆ 30.0     ┆ 30.0               ┆ 0                       ┆ true              │
└─────────┴──────────┴────────────────────┴─────────────────────────┴───────────────────┘
(row 0: private, capped 60 -> 48; row 1: public, cap is 60, unchanged;
 row 2: private, 30 is already under the 48 cap, unchanged — `term_cap_by
 _sector_path` names which arm ran: 0 = private, 1 = public)

=== 2. Plain Loop: ExtendTermToFit ===
shape: (2, 2)
┌───────────┬──────────┐
│ principal ┆ term_cap │
│ ---       ┆ ---      │
│ f64       ┆ f64      │
╞═══════════╪══════════╡
│ 100000.0  ┆ 40.0     │
│ 20000.0   ┆ 12.0     │
└───────────┴──────────┘
(started both at term_cap=12; row 0's instalment was 100000/12=8333,
 over the 2500 ceiling, so the loop ran 28 real iterations to reach 40
 (100000/40=2500, exactly at the ceiling, stops); row 1's instalment was
 20000/12=1667, already under 2500, so should_continue was false on the
 very FIRST check — 0 iterations ran, a real early exit)

=== 3. Nested: SearchForViableTerm = Loop(Branch(steps1, Loop(steps2))) ===
shape: (2, 4)
┌─────────────┬───────────┬──────────┬────────────────┐
│ high_income ┆ principal ┆ term_cap ┆ is_high_income │
│ ---         ┆ ---       ┆ ---      ┆ ---            │
│ bool        ┆ f64       ┆ f64      ┆ bool           │
╞═════════════╪═══════════╪══════════╪════════════════╡
│ true        ┆ 100000.0  ┆ 48.0     ┆ true           │
│ false       ┆ 100000.0  ┆ 42.0     ┆ false          │
└─────────────┴───────────┴──────────┴────────────────┘
(both start term_cap=12, outer floor=40. high_income=true takes FastTrack
 — itself a Loop, 3 inner micro-steps of +4 = +12/outer iteration: 12 ->
 24 -> 36 -> 48, three outer iterations, overshooting the 40 floor to 48
 because it jumps in steps of 12; high_income=false takes SlowTrack, a
 plain single step, +3/outer iteration: 12 -> 15 -> ... -> 39 -> 42, ten
 outer iterations, overshooting by less because its step is smaller.
 `AdjustmentStrategy`'s own `_path` fired correctly inside every one of
 those outer iterations — see "what this playground could NOT show" below
 for why it cannot also be `.emit()`-ed at the outer pipeline's level.)

assert_equivalent(pipeline, frame): OK — interpreted, stepped, fused and
score() all agree, exactly, on every row in every section above (each
section calls it; no output beyond "OK" because there is nothing to
report when it passes).

=== 4. Serving it ===
See the module docstring's `decider2 serve` command above — `pipeline`
below (the nested demo, section 3) is exactly what `decider2 serve` will
find and load.

**What this playground could NOT show, and why (found while building it,
not designed in advance).** `AdjustmentStrategy_path` cannot be `.emit()`-ed
from `SearchForViableTerm`'s own pipeline: `AdjustmentStrategy` is the
INTERIOR of `SearchForViableTerm`'s body, and `Loop`'s own generated
function only returns `carries` — `adjustment_strategy_path` is computed
and consumed entirely inside the generated `while`, once per iteration, and
never surfaces as one of `Loop`'s own outputs. `X_path` is only reachable
where `X` is a Branch used DIRECTLY in the outer pipeline (section 1) —
never a Branch nested inside a Loop's body (section 3). This is a genuine,
previously-untested limitation of the `<name>_path` convention once
nesting is involved; see this agent's final report.

**A second thing combining all three sections into ONE `pipeline` ran
into**, also found here rather than anywhere designed in advance: sections
1 and 2/3 each declare their own leaf called `term_cap`, meaning something
different in each (a sector cap vs. a loop carry). Doc 03 §2.1's forward-
reference check correctly refuses to compose them into one pipeline (§1's
module would read a leaf that §2's/§3's module produces later) — the same
"same name, same meaning" rule this whole design rests on, applied to a
mistake this playground itself first made. `pipeline` below is therefore
just section 3, and `_run_and_print()` builds each section as its own
independent pipeline.
--------------------------------------------------------------------------

**What `carries` means, concretely**: `ExtendTermToFit`'s only carry is
`term_cap` — it is the loop body's own input at the START of an iteration
(the value from the row, or the previous iteration) and the loop body MUST
produce a new `term_cap` by the END of the iteration (a self-read
waterfall, one level up from doc 03 §3.2's ordinary one). `SearchForViable
Term` carries `term_cap` too, for the same reason, one level further out —
Loop nests by having an outer Loop's body BE another Branch/Loop, not by
any special syntax.

**Why `max_iterations` is required, not optional**: once `ExtendTermToFit`
compiles, it is a real `while` loop in machine code (doc 03 §8.3) — nothing
outside that compiled function can interrupt it mid-flight. An unbounded
loop over malformed or adversarial data would simply hang the process. The
bound is the only thing standing between "a policy rule" and "a denial-of-
service." `Loop(...)` enforces this the loudest way Python has: omitting
`max_iterations` is a bare `TypeError`, before a single line of `Loop`'s
own code runs.

**A limitation this playground works around, not something Loop hides**:
`FastTrackBump`'s own `should_continue` reads only `loop_idx` (not the
outer `term_cap`), because a `Loop`'s `should_continue` can only read a
name that is either a genuine leaf, `loop_idx`, or one of `carries` — never
a body output that ISN'T a carry, since that value does not exist yet on
the very first check (`decider2.graph.control_flow.loop` raises a build
error for this — found while building this playground, not designed in
advance). See this agent's final report for the rest of what doc 03
§8.2/§8.3 left unsaid.
"""
from __future__ import annotations

import polars as pl

from decider2 import Branch, Loop, flow, module, param, step

# ---------------------------------------------------------------------------
# 1. A plain Branch — doc 03 §8.2's own worked example, with real numbers.
# ---------------------------------------------------------------------------


def is_private_sector(sector: str, private: str = param("private")) -> bool:
    """Condition step -> bool. Doc 05 §1.5: a str-typed leaf's literal
    comparison must go through a declared str param() — see
    tests/test_control_flow.py's module docstring for the full note."""
    return sector == private


@step(output="term_cap_by_sector")
def cap_for_private(term_cap: float, private_cap: float = param(48.0, ge=6, le=60)) -> float:
    """Private-sector applicants: capped at 48 months."""
    return min(term_cap, private_cap)


@step(output="term_cap_by_sector")
def cap_for_public(term_cap: float, public_cap: float = param(60.0, ge=6, le=60)) -> float:
    """Public-sector applicants: capped at 60 months."""
    return min(term_cap, public_cap)


TermCapBySector = Branch(
    is_private_sector,
    module(cap_for_private),
    module(cap_for_public),
    modifies=["term_cap_by_sector"],
    name="term_cap_by_sector",
)

# ---------------------------------------------------------------------------
# 2. A plain Loop — real early exit, real bound.
# ---------------------------------------------------------------------------


def instalment_fits(term_cap: float, loop_idx: int, principal: float, ceiling: float = param(2500.0)) -> bool:
    """should_continue: (carried..., loop_idx) -> bool (doc 03 §8.3).
    Keeps extending the term while the (simplified, no-interest)
    instalment is still over the ceiling."""
    return (principal / term_cap) > ceiling and loop_idx < 1000


@step(output="term_cap")
def extend_one_month(term_cap: float, loop_idx: int) -> float:
    """The loop body: one more month of term, self-reading its own prior
    value (doc 03 §3.2's waterfall, one level up)."""
    return term_cap + 1.0 + 0.0 * loop_idx


ExtendTermToFit = Loop(
    instalment_fits,
    module(extend_one_month),
    carries=["term_cap"],
    max_iterations=511,  # REQUIRED — doc 03 §8.3
    name="extend_term_to_fit",
)

# ---------------------------------------------------------------------------
# 3. Nested: Loop(Branch(steps1, Loop(steps2)))
# ---------------------------------------------------------------------------


def is_high_income(high_income: bool) -> bool:
    """Branch condition — a plain bool leaf, no str-boundary workaround
    needed."""
    return high_income


def fast_track_should_continue(loop_idx: int, micro_steps: float = param(3.0)) -> bool:
    """The INNER Loop's own should_continue: a small, fixed number of
    micro-steps per invocation (doc 03 §2's reserved `loop_idx`) — it
    cannot read the outer `term_cap` at all (see this module's own
    docstring for why: a body output that is not a carry has no value on
    should_continue's very first check, and `term_cap` is the OUTER loop's
    carry, not this inner Loop's own)."""
    return loop_idx < micro_steps


@step(output="term_cap")
def fast_bump(term_cap: float, loop_idx: int, jump: float = param(4.0)) -> float:
    """One inner micro-step: +4 months. 3 micro-steps per FastTrackBump
    call = +12 months per OUTER iteration."""
    return term_cap + jump + 0.0 * loop_idx


FastTrackBump = Loop(
    fast_track_should_continue,
    module(fast_bump),
    carries=["term_cap"],
    max_iterations=3,
    name="fast_track_bump",
)  # <-- this IS "Loop(steps2)": a Loop used as one arm of a Branch


@step(output="term_cap")
def slow_bump(term_cap: float, loop_idx: int, step_size: float = param(3.0)) -> float:
    """The other arm: a plain, single step (+3 months) — no inner loop."""
    return term_cap + step_size + 0.0 * loop_idx


AdjustmentStrategy = Branch(
    is_high_income,
    FastTrackBump,  # arm for True — a Loop
    module(slow_bump),  # arm for False — a plain step
    modifies=["term_cap"],
    name="adjustment_strategy",
)  # <-- "Branch(steps1, Loop(steps2))"


def term_still_short(term_cap: float, loop_idx: int, floor: float = param(40.0)) -> bool:
    """The OUTER Loop's should_continue — reads `term_cap`, its own carry,
    never the inner Loop's `loop_idx` or micro-step machinery."""
    return term_cap < floor and loop_idx < 1000


SearchForViableTerm = Loop(
    term_still_short,
    AdjustmentStrategy,
    carries=["term_cap"],
    max_iterations=511,
    name="search_for_viable_term",
)  # <-- "Loop(Branch(steps1, Loop(steps2)))" in full

# ---------------------------------------------------------------------------
# The pipeline `decider2 serve` finds (doc 03 §6's convention: a module-
# level `pipeline = ...`). It is the NESTED demo alone, standalone: the
# three sections above each build their own leaf named `term_cap`
# (§2.1's ordinary "same name, same meaning" rule), and §1's and §2's
# `term_cap`s mean genuinely different things (a sector cap vs. a loop
# carry) — chaining all three into one pipeline would make §1's module
# read a leaf that §2's/§3's module later produces, which doc 03 §2.1's
# forward-reference check correctly refuses (found while writing this
# playground; see this agent's report). `_run_and_print()` below runs
# each section as its own independent pipeline instead, which is also
# the more honest demo: three separate rules, not one entangled one.
# ---------------------------------------------------------------------------

pipeline = flow(is_high_income, SearchForViableTerm)


def _run_and_print() -> None:
    from decider2.testing import assert_equivalent

    print("=== 1. Plain Branch: TermCapBySector ===")
    sector_frame = pl.DataFrame(
        {"sector": ["private", "public", "private"], "term_cap": [60.0, 60.0, 30.0]}
    )
    sector_pipeline = flow(is_private_sector, TermCapBySector).emit("term_cap_by_sector_path")
    print(sector_pipeline.apply(sector_frame, mode="interpreted"))
    assert_equivalent(sector_pipeline, sector_frame)

    print()
    print("=== 2. Plain Loop: ExtendTermToFit ===")
    loop_frame = pl.DataFrame({"principal": [100_000.0, 20_000.0], "term_cap": [12.0, 12.0]})
    loop_pipeline = flow(ExtendTermToFit)
    print(loop_pipeline.apply(loop_frame, mode="interpreted"))
    assert_equivalent(loop_pipeline, loop_frame)

    print()
    print("=== 3. Nested: SearchForViableTerm = Loop(Branch(steps1, Loop(steps2))) ===")
    nested_frame = pl.DataFrame(
        {"high_income": [True, False], "principal": [100_000.0, 100_000.0], "term_cap": [12.0, 12.0]}
    )
    nested_pipeline = flow(is_high_income, SearchForViableTerm)
    print(nested_pipeline.apply(nested_frame, mode="interpreted"))
    assert_equivalent(nested_pipeline, nested_frame)
    print()
    print("assert_equivalent(pipeline, frame): OK for all three sections above.")


if __name__ == "__main__":
    _run_and_print()
