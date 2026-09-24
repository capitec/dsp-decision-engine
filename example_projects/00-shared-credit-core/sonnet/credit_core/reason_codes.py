"""`core.reason_codes` -- the decline reason taxonomy (spec 00 §6.18).

A registry, not a calculation: every reason code the Bank may communicate,
its severity rank (1 = most severe), its regulatory classification and its
effective dates. Given the set of codes a flow fired, this module ranks
them and picks the primary one communicated to the client. Dropping the
registry during a port is the exact failure this capability exists to make
impossible (00 §6.18 "Hard part").
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

from decider import step


class ReasonCode(NamedTuple):
    code: int
    severity_rank: int          # 1 = most severe
    description: str
    is_regulatory: bool         # a regulatory-mandated reason vs. a discretionary one


class ReasonCodeRegistry:
    """A versioned set of `ReasonCode`s. Testable standalone (§7.5): no pipeline needed."""

    def __init__(self, version: str, codes: Sequence[ReasonCode]):
        self.version = version
        by_code = {c.code: c for c in codes}
        if len(by_code) != len(codes):
            raise ValueError(f"reason registry {version}: duplicate codes")
        self._by_code = by_code

    def __contains__(self, code: int) -> bool:
        return code in self._by_code

    def rank(self, fired: Sequence[int]) -> tuple[list[int], int | None]:
        """Every fired code the registry knows, most severe first, and the primary.

        Unknown codes are dropped rather than silently kept unranked --
        raise instead, since a code the registry does not recognise is
        exactly the defect this capability exists to prevent.
        """
        unknown = [c for c in fired if c not in self._by_code]
        if unknown:
            raise LookupError(f"reason codes not in registry {self.version}: {sorted(set(unknown))}")
        ordered = sorted(set(fired), key=lambda c: (self._by_code[c].severity_rank, c))
        primary = ordered[0] if ordered else None
        return ordered, primary

    def resolve_step(self):
        """A decider step: reads `decline_reason_codes` (as fired), writes the ranked
        list back under the same name plus `primary_reason_code` and `reason_registry_version`.
        """
        registry = self

        def rank_reasons(decline_reason_codes: list[int]) -> tuple[list[int], int | None, str]:
            # `is None`, never a bare truthiness test: a non-empty list input arrives as a
            # numpy array in this step's row loop, and `array or []` raises "truth value of
            # an array with more than one element is ambiguous".
            fired = [] if decline_reason_codes is None else decline_reason_codes
            ordered, primary = registry.rank(fired)
            return ordered, primary, registry.version

        rank_reasons.__name__ = "rank_reasons"
        return step(rank_reasons, outputs=("decline_reason_codes", "primary_reason_code", "reason_registry_version"))
