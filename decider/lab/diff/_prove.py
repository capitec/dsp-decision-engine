from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from decider.lab.diff._structure import StructureDiff, structure
from decider.ir import step_map


class SmtEvidence(BaseModel):
    """What a z3 proof covered: the steps compared, what was modelled, and a counterexample if they differ.

    `counterexample` is an input row (None is a null) for the step at `step`, on which the two versions
    write different outputs. The step is judged on its own, so the flow around it may never produce that row.
    """

    model_config = ConfigDict(frozen=True)

    modelled: str
    steps: tuple[str, ...]
    step: str | None = None
    counterexample: dict[str, Any] | None = None


class Proof(BaseModel):
    """Whether two pipelines provably compute the same thing, and by which tier.

    `tier="ir_hash"`: equal IR content hashes. `tier="smt"`: the flows differ only in tree and decision-table
    steps, and z3 found no input on which those steps disagree (`identical=True`). Otherwise `identical` is None, not False:
    nothing here proves them the same or different. If z3 found an input on which a step differs, it is
    `evidence.counterexample` (the step alone, which the flow may never reach); else `evidence` shows what
    differs and `reason` why z3 wasn't used.
    """

    model_config = ConfigDict(frozen=True)

    tier: Literal["ir_hash", "smt", "none"]
    identical: bool | None
    evidence: StructureDiff | SmtEvidence | None = None
    reason: str | None = None


def prove(old: Any, new: Any) -> Proof:
    """Prove two pipelines identical, or say what differs.

    Trees and decision tables that changed are compared with z3 when `decider[prove]` is installed.

    Example::

        diff.prove(old, new)   # Proof(tier="ir_hash", identical=True) for a pure refactor
    """
    s = structure(old, new)
    return Proof(tier="ir_hash", identical=True) if s.identical else _smt(old, new, s)


def _smt(old: Any, new: Any, s: StructureDiff) -> Proof:
    def fallback(reason: str) -> Proof:
        return Proof(tier="none", identical=None, evidence=s, reason=reason)

    try:
        import z3  # noqa: F401  (the extra may be missing, even when _smt was imported earlier)
        from decider.lab.diff import _smt as smt
    except ImportError:
        return fallback("install decider[prove] to compare trees and decision tables with z3")
    if s.params:
        return fallback("a param declaration differs, and z3 compares only params-free trees and tables")
    if s.added or s.removed or s.renamed or s.moved or s.edges or not s.changed_steps:
        return fallback("the pipelines differ in more than the contents of steps, which z3 can't compare")
    before, after = step_map(old), step_map(new)
    proven, why = [], None
    for change in s.changed_steps:
        path = change.ref.path
        try:
            if path not in before or path not in after:
                raise smt.Unencodable("no step at this path in both versions")
            row = smt.differ(before[path], after[path])
        except smt.Unencodable as e:
            why = why or f"{path}: {e}"
            continue
        if row is not None:
            return Proof(tier="smt", identical=None,
                         evidence=SmtEvidence(modelled=smt.MODELLED, steps=(path,), step=path, counterexample=row),
                         reason=f"{path}: z3 found an input on which the step differs; it judges the step alone, "
                                "so the flow around it may never produce that input")
        proven.append(path)
    if why:
        return fallback(why)
    return Proof(tier="smt", identical=True, evidence=SmtEvidence(modelled=smt.MODELLED, steps=tuple(proven)))
