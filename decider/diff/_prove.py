from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from decider.diff._structure import StructureDiff, structure


class Proof(BaseModel):
    """Whether two pipelines provably compute the same thing.

    `identical=True` only when their IR content hashes are equal (`tier="ir_hash"`).
    Otherwise `identical` is None, not False: nothing here proves them different
    or the same, and `evidence` shows what differs.
    """

    model_config = ConfigDict(frozen=True)

    tier: Literal["ir_hash", "none"]
    identical: bool | None
    evidence: StructureDiff | None = None


def prove(old: Any, new: Any) -> Proof:
    """Prove two pipelines identical, or say what differs.

    Example::

        diff.prove(old, new)   # Proof(tier="ir_hash", identical=True) for a pure refactor
    """
    s = structure(old, new)
    return Proof(tier="ir_hash", identical=True) if s.identical else Proof(tier="none", identical=None, evidence=s)
