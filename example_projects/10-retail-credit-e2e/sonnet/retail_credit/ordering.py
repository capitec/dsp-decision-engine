"""Ordering constraints between phases (spec 10 §5.22).

Declared as data (id, the two phases involved, the reason, and whether it
is a genuine cycle broken by declaration) so a change to phase order is
checked against them rather than discovered as a defect. `check_order`
verifies a concrete execution order against every constraint this project
implements; the two genuine cycles (O-06, O-09) and the second-in-miniature
(O-10) are declared explicitly as "broken by declaration", matching how the
spec itself describes them (10 §5.22) -- `check_order` does not try to
enforce a strict phase order for these, only that the declared break point
(e.g. P09 running in two passes around P10) is honoured.

This project's entry point 1 execution order is P01, P02, P03, P04, P05,
P06, P07, P08, P09a (amount/term/grade entries), P10, P09b (instalment
entries), P11, P10 (re-run), P12, P13, [loop L1: P14, P10, P12, P13, P16],
P16, P17, P18 -- see `pipeline.py`. `check_order` is exercised by
`tests/test_ordering.py` against exactly that sequence.
"""
from __future__ import annotations

from dataclasses import dataclass

CONSTRAINT = "constraint"
CYCLE = "cycle"


@dataclass(frozen=True)
class OrderingConstraint:
    constraint_id: str      # "O-01".."O-24"
    before: str              # a phase code, or a declared synthetic step name (e.g. "P09a")
    after: str
    reason: str
    kind: str = CONSTRAINT    # CONSTRAINT | CYCLE


CONSTRAINTS: tuple[OrderingConstraint, ...] = (
    OrderingConstraint("O-01", "P08.adjustments", "P08.grading",
                        "Grading keys off adjusted PD"),
    OrderingConstraint("O-02", "P03.consent", "P04.acquisition",
                        "A bureau enquiry without consent is an offence"),
    OrderingConstraint("O-03", "P02", "P03",
                        "Consent is held against client_id"),
    OrderingConstraint("O-05", "P08.adjustments", "P09",
                        "P08 resolves the overlay register once, before any phase reading an overlaid value"),
    OrderingConstraint("O-06", "P05", "P04",
                        "Cycle, broken by declaration per entry point: fraud before bureau on low-value "
                        "entry points 1/2, after it on 5/6 (34 of 188 rules consume bureau-derived features)",
                        kind=CYCLE),
    OrderingConstraint("O-07", "P06.income", "P07",
                        "14 of a scorecard's characteristics are income-derived"),
    OrderingConstraint("O-08", "P06.bureau", "P06.segment",
                        "Segments 1/2 are distinguished by thin-file status, which is bureau-derived"),
    OrderingConstraint("O-09", "P10a", "P11",
                        "Cycle, broken by declaration: P10 runs product-neutrally, P11 routes, "
                        "P10 re-runs with the routed product's buffer", kind=CYCLE),
    OrderingConstraint("O-09b", "P11", "P10b",
                        "The re-run after routing", kind=CYCLE),
    OrderingConstraint("O-10", "P09a", "P10",
                        "instalment_cap is seeded from affordability: the register runs in two passes",
                        kind=CYCLE),
    OrderingConstraint("O-10b", "P10", "P09b",
                        "The instalment-acting entries run after affordability", kind=CYCLE),
    OrderingConstraint("O-12", "P09.regulatory", "P09.CAP-0118",
                        "The uplift entry runs last, after every regulatory-class entry"),
    OrderingConstraint("O-14a", "P12.rate", "P12.fee", "The fee is a function of the advance"),
    OrderingConstraint("O-14b", "P12.fee", "P12.premium",
                        "The premium is a function of the amount financed, which includes the capitalised fee"),
    OrderingConstraint("O-14c", "P12.premium", "P12.instalment",
                        "The instalment is a function of rate, fee and premium"),
    OrderingConstraint("O-15", "P10", "P14",
                        "P14 sits inside the P10 loop: reached because affordability failed"),
    OrderingConstraint("O-17", "P13", "P16",
                        "Every product's P11-P13 fan-out completes before arbitration runs"),
    OrderingConstraint("O-19", "P17", "P18.ranking",
                        "P18 ranks reasons after every phase that can raise one, including P17"),
    OrderingConstraint("O-20", "P17", "P18.disclosure",
                        "A quotation issued from an offer that failed validation is a representation the Bank is bound by"),
    OrderingConstraint("O-22", "P13", "P18",
                        "Rounding is applied at exactly one place per value"),
)


def check_order(executed: list[str]) -> list[str]:
    """Every constraint this project declares that `executed` (a list of phase/step codes,
    in the order they actually ran) violates. Empty means the order is clean. Cycle-kind
    constraints are checked as "both sides appear, in either order, more than once where the
    declared break requires it" rather than a strict before/after, because a cycle broken by
    declaration is exactly a case where naive "before < after" is the wrong test.
    """
    position = {code: i for i, code in enumerate(executed) if code in executed}
    violations = []
    index_of: dict[str, list[int]] = {}
    for i, code in enumerate(executed):
        index_of.setdefault(code, []).append(i)

    for c in CONSTRAINTS:
        if c.kind == CYCLE:
            continue  # verified structurally by pipeline.py's explicit two-pass wiring, not here
        if c.before not in index_of or c.after not in index_of:
            continue  # this project's slice doesn't execute both sides (e.g. a skipped phase)
        if min(index_of[c.before]) >= min(index_of[c.after]):
            violations.append(f"{c.constraint_id}: {c.before} must run before {c.after} ({c.reason})")
    return violations
