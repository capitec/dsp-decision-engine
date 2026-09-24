"""09-H's one servable capability: replay a captured decision on demand.

This project is not a decision flow (spec 09 §1: "no applicants, no
scorecards, ... no outcome of its own"). Its real capabilities --
explanation, diff, swap-set, the overlay register, dead-logic coverage --
return deeply nested, tree-shaped documents (a cap chain, a rule-by-rule
attribution table, a diff summary): exactly the shapes 00/01/03/05's own
NOTES.md document `decider`'s typed step model rejects at the framework
boundary (`list[struct]`/`list[list[T]]` outputs crash result
materialisation -- see those projects' "Framework friction" sections).
Rather than force them through steps they would fight, or invent a
flat-parallel-list encoding purely to satisfy `decider build`, this
project's servable pipeline is deliberately the one capability whose inputs
and outputs are plain scalars: replay a decision by id, report the verdict.
Everything else this harness builds (`governance/explain.py`, `diff.py`,
`swapset.py`, `overlay_register.py`, `deadlogic.py`) is a tested Python
library, called directly -- see `tests/` and NOTES.md "What I built"/
"Framework friction" for why that is the honest shape for this project, not
a shortcut around `decider`.
"""
from __future__ import annotations

from decider import param, step

from governance.replay import replay_by_id


def run_replay(flow_code: str, decision_id: str, mode: str = param("interpreted")) -> tuple[str, int, str]:
    """Replays one previously captured decision (spec 09 §5.1) and reports the verdict.

    Returns `(verdict, divergence_count, diverged_fields)` -- `diverged_fields` is a
    comma-joined string, not a `list[str]`, deliberately: keeping every output here a bare
    scalar keeps this pipeline typed the same simple way on both `.score()` and `.run()`.
    """
    verdict = replay_by_id(flow_code, decision_id, mode=mode)
    diverged = ",".join(d.field for d in verdict.divergences)
    return verdict.verdict, len(verdict.divergences), diverged


replay_step = step(run_replay, outputs=("replay_verdict", "divergence_count", "diverged_fields"))


def build():
    return replay_step
