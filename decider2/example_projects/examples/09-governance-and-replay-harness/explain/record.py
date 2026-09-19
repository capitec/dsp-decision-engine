"""The one underlying record. Three audiences are projections of it, never
three documents.

Spec §5.2 gives three renderings that "differ in what they may contain". The
obvious implementation - three templates over the same dict - fails the moment
a new field is added, because the new field is in all three or none, and the
person adding it is an engineer on a flow team who has never met an ombud.

So the record carries CLASSIFICATION per field (explain/disclosure.py) and a
rendering is a projection through a disclosure level. Adding a field cannot
leak it, because an unclassified field does not build.
"""

from __future__ import annotations

from datetime import date, datetime

from manifest.model import CapabilityPin
from replay.pin_resolution import Assignment, Received


class GateEvaluation:
    """Item 14: EVALUATED, not only fired."""
    gate_id: str
    order: int
    evaluated: bool
    outcome: bool
    values_tested: dict           # the actual values, classified
    branch_path: int              # the int64 immediate, doc 04 §4.1


class RuleVerdict:
    rule_id: str                  # "CAP-0210" - stable, never positional (item 2)
    sequence: int
    applicable: bool
    fired: bool
    bound: bool                   # for a waterfall rule: did it actually move the ceiling
    coincident_with: str | None   # spec 03 §5.5: two rules reducing to the same value
    why_not: str | None           # "not applicable: existing client" vs "evaluated, did not bind"
    owner: str
    effective_from: date
    effective_to: date | None
    reason_codes_raised: tuple[int, ...]


class ChainLink:
    """Item 15. The ordered chain, not the final value with a note.

    This is doc 04 §5.1's version chain, one link per boundary crossing, with
    the producer attached. The harness does not build it - `attest()` emits it,
    because `term_cap@*` (doc 03 §7) is already the framework's own answer to
    "every version as its own column". The ONLY thing the harness adds is the
    cause string, and the cause string is the rule's `holds` sentence.
    """
    value: int                    # scaled int64 cents (doc 03 §1.2). Never float for money.
    producer: str                 # module instance name, never a position
    cause: str                    # the clause sentence, in force at decision_date
    kind: "Literal['seed', 'rule', 'overlay', 'rounding', 'statutory_ceiling']"
    artefact_ref: str             # rule id, or overlay id + approval ref


class ScoreContribution:
    """Item 16. A required OUTPUT, not a diagnostic (00 §6.10)."""
    characteristic: str
    value: object
    bin_label: str                # including the null bin, which is a bin and not an error
    points: float                 # signed
    rank: int


class DecisionRecord:
    decision_id: str
    flow: str
    governance_grade: "Literal['A', 'B', 'C']"     # from the attestation, printed on every rendering
    decision_date: date
    computed_at: datetime

    inputs: dict[str, Received]
    gates: tuple[GateEvaluation, ...]
    rules: tuple[RuleVerdict, ...]
    tree_paths: dict[str, tuple[str, ...]]         # tree_id -> ordered node_keys
    chains: dict[str, tuple[ChainLink, ...]]       # "amount_cap" -> the whole waterfall
    contributions: tuple[ScoreContribution, ...]
    cells_read: tuple[tuple[str, tuple, object], ...]   # (table_version, coords, value)
    overlays: "OverlayApplication"                 # adjusted AND unadjusted, side by side (§5.14.5)
    assignments: dict[str, Assignment]
    capabilities: tuple[CapabilityPin, ...]

    outcome_code: int
    decline_reason_codes: tuple[int, ...]          # severity order
    primary_reason_code: int
    reason_registry_version: str                   # the wording IN FORCE AT decision_date, not today's

    def wording(self, code: int, language: str) -> str:
        """Spec §11.8: Compliance adds 40 codes and reclassifies 12; every
        historical explanation must still render with the wording in force at
        its own decision date. So this resolves through the registry version
        pinned on the record, never the current registry."""
        pass
