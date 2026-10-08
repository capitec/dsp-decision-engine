"""Effective-dated artefacts: pick the version in force on the record's own date, never on today's."""
from __future__ import annotations

from datetime import date
from typing import NamedTuple, Sequence

from decider import allow_fallback, step


class EffectiveVersion(NamedTuple):
    version_id: str
    effective_from: date
    effective_to: date | None = None  # None: still in force


class EffectiveDatedSet:
    """The version history of one artefact family.

    Example::

        norms = EffectiveDatedSet("norms", [EffectiveVersion("v1", date(2025, 1, 1), date(2026, 1, 1)),
                                            EffectiveVersion("v2", date(2026, 1, 1))])
        norms.resolve(date(2025, 6, 1)).version_id   # "v1"
        flow(norms.resolver_step(), lookup_table)    # writes `norms_version` from `decision_date`
    """

    def __init__(self, family: str, versions: Sequence[EffectiveVersion]):
        self.family = family
        self.versions = tuple(sorted(versions, key=lambda v: v.effective_from))
        for prev, nxt in zip(self.versions, self.versions[1:]):
            if prev.effective_to != nxt.effective_from:
                raise ValueError(f"{family}: {prev.version_id} ends {prev.effective_to}, "
                                 f"{nxt.version_id} starts {nxt.effective_from}: gap or overlap")

    def resolve(self, as_of: date) -> EffectiveVersion:
        """The version in force on `as_of`; `LookupError` if none is."""
        for v in self.versions:
            if v.effective_from <= as_of and (v.effective_to is None or as_of < v.effective_to):
                return v
        raise LookupError(f"no version of {self.family!r} is in force on {as_of}")

    def resolver_step(self, *, date_field: str = "decision_date", output: str | None = None):
        """A step that reads `date_field` and writes the version id (default output `<family>_version`)."""

        @allow_fallback
        def resolve_version(decision_date: date) -> str:
            return self.resolve(decision_date).version_id

        resolve_version.__name__ = f"resolve_{self.family.replace('.', '_')}_version"
        s = step(resolve_version, output=output or f"{self.family.replace('.', '_')}_version")
        return s if date_field == "decision_date" else s.relabel(reads={"decision_date": date_field})
