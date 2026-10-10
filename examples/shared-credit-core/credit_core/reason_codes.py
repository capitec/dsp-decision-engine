"""The decline reason taxonomy: severity ranking and the one reason communicated to the client."""
from __future__ import annotations

from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, model_validator

from decikit import ConfigurableStep, allow_fallback


class Code(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    code: int
    severity_rank: int  # 1 is the most severe
    description: str
    is_regulatory: bool


def rank_codes(fired: Sequence[int], severity: tuple[tuple[int, int], ...], version: str) -> tuple[list[int], int | None]:
    rank = dict(severity)
    unknown = sorted({c for c in fired if c not in rank})
    if unknown:
        raise LookupError(f"reason codes not in registry {version}: {unknown}")
    ordered = sorted(set(fired), key=lambda c: (rank[c], c))
    return ordered, (ordered[0] if ordered else None)


@allow_fallback
def rank_reasons(decline_reason_codes: list[int], severity: tuple[tuple[int, int], ...], version: str,
                 ) -> tuple[list[int], int | None, str]:
    ordered, primary = rank_codes(decline_reason_codes, severity, version)
    return ordered, primary, version


class ReasonCodes(ConfigurableStep):
    """A versioned set of reason codes that ranks `decline_reason_codes` in place.

    It writes `primary_reason_code` (the most severe fired code) and `reason_registry_version`. A code outside the
    set raises.

    Example::

        codes = ReasonCodes(version="v1", codes=[Code(code=7, severity_rank=1, description="Sanctioned party", is_regulatory=True),
                                                 Code(code=9, severity_rank=2, description="Low score", is_regulatory=False)])
        codes.rank([9, 7])   # ([7, 9], 7)
        flow(fired_step, codes)
    """

    type: Literal["reason_codes"] = "reason_codes"
    name: str = "rank_reasons"
    version: str
    codes: tuple[Code, ...]

    @model_validator(mode="after")
    def _codes_unique(self) -> ReasonCodes:
        if len({c.code for c in self.codes}) != len(self.codes):
            raise ValueError(f"reason registry {self.version}: duplicate codes")
        return self

    def __contains__(self, code: int) -> bool:
        return any(c.code == code for c in self.codes)

    def _severity(self) -> tuple[tuple[int, int], ...]:
        return tuple((c.code, c.severity_rank) for c in self.codes)

    def rank(self, fired: Sequence[int]) -> tuple[list[int], int | None]:
        """The distinct fired codes, most severe first, and the primary one. A code outside the set raises."""
        return rank_codes(fired, self._severity(), self.version)

    def to_ir(self, ctx: Any) -> Any:
        return ctx.call(self, rank_reasons, values={"severity": self._severity(), "version": self.version},
                        outputs=("decline_reason_codes", "primary_reason_code", "reason_registry_version"))
