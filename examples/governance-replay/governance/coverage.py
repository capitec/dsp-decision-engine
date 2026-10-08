"""Which rules decisions actually used: dead logic, dominant logic and the cells read."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from decider.lab import checks

from governance.flows import Flow

DOMINANT = 0.40


@dataclass(frozen=True)
class Coverage:
    """Rule hits over a population. `dead` rules never bound, `dominant` ones bound on more than 40% of
    decisions, `unknown` ids bound but are not in the flow's rule list. `unreachable` are branches decider's
    static checks prove can never run, whatever the traffic; `cells` counts distinct table cells read per field."""

    decisions: int
    hits: dict[str, int]
    dead: tuple[str, ...]
    dominant: tuple[str, ...]
    unknown: tuple[str, ...]
    unreachable: tuple[str, ...]
    cells: dict[str, int]


def coverage(flow: Flow, responses: list[dict], version: str | None = None) -> Coverage:
    """Count rule hits in `responses` (decision records) against the rules of `version`.

        coverage(LOAN, [e.response for e in store.all("loan")]).dead
    """
    store = flow.store()
    version = version or store.latest_version()
    universe = flow.rules(store.read(version).config)
    fired = Counter(rule for r in responses for rule in flow.fired(r))
    n = len(responses)
    report = checks.run(flow.pipeline(str(version)), ["unreachable_branch"])
    cells = {f: len({r[f] for r in responses if r.get(f)}) for f in sorted(responses[0]) if f.endswith("_cell_id")}
    return Coverage(
        n, {rule: fired[rule] for rule in sorted(universe)},
        tuple(sorted(universe - fired.keys())),
        tuple(sorted(rule for rule in universe if n and fired[rule] / n > DOMINANT)),
        tuple(sorted(r for r in fired.keys() - universe if r)),
        tuple(f"{f.code} {f.step.path}" for f in report.findings), cells)
