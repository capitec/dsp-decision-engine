"""The decline reason taxonomy: severity ranking and the one reason communicated to the client."""
from __future__ import annotations

from typing import NamedTuple, Sequence

from decider import allow_fallback, step


class ReasonCode(NamedTuple):
    code: int
    severity_rank: int  # 1 is the most severe
    description: str
    is_regulatory: bool


class ReasonCodeRegistry:
    """A versioned set of reason codes.

    Example::

        registry = ReasonCodeRegistry("v1", [ReasonCode(7, 1, "Sanctioned party", True), ReasonCode(9, 2, "Low score", False)])
        registry.rank([9, 7])   # ([7, 9], 7)
    """

    def __init__(self, version: str, codes: Sequence[ReasonCode]):
        self.version = version
        self._by_code = {c.code: c for c in codes}
        if len(self._by_code) != len(codes):
            raise ValueError(f"reason registry {version}: duplicate codes")

    def __contains__(self, code: int) -> bool:
        return code in self._by_code

    def rank(self, fired: Sequence[int]) -> tuple[list[int], int | None]:
        """The distinct fired codes, most severe first, and the primary one. A code outside the registry raises."""
        unknown = sorted({c for c in fired if c not in self._by_code})
        if unknown:
            raise LookupError(f"reason codes not in registry {self.version}: {unknown}")
        ordered = sorted(set(fired), key=lambda c: (self._by_code[c].severity_rank, c))
        return ordered, (ordered[0] if ordered else None)

    def resolve_step(self):
        """A step that ranks `decline_reason_codes` in place and writes `primary_reason_code` and `reason_registry_version`."""

        @allow_fallback
        def rank_reasons(decline_reason_codes: list[int]) -> tuple[list[int], int | None, str]:
            ordered, primary = self.rank(decline_reason_codes)
            return ordered, primary, self.version

        return step(rank_reasons, outputs=("decline_reason_codes", "primary_reason_code", "reason_registry_version"))
