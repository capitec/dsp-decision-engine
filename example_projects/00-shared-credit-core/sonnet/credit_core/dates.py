"""`core.dates` -- effective-dated artefact selection (spec 00 §6.19, §7.3).

Every versioned artefact in the library (tax tables, expense norms, fee
caps, rate cards, scorecards, appetite grids, the reason taxonomy) is
selected by `decision_date`, never by "today" (09 §5.15 item 4). Addendum
A8 extends this to dates other than `decision_date` (e.g. `knowledge_date`)
and to per-assessment major-version choice, so the resolver is parameterised
on which date field and which artefact family, not hardwired to one of
either.

Design: an `EffectiveDatedSet` is a plain, standalone-testable Python
object (§7.5 -- no pipeline, no frame, no database needed to test it). A
capability that reads an effective-dated table builds a *resolver step*
from one, which writes the version id in force for each record's own date.
Downstream, the table lookup itself ANDs that version id into its match
expression, so one table document can hold every historical version's rows
side by side: an old record can never see a new row, and a new version
never overwrites an old one in place (09 §5.15 item 5).
"""
from __future__ import annotations

from datetime import date
from typing import NamedTuple, Sequence

from decider import step


class EffectiveVersion(NamedTuple):
    version_id: str
    effective_from: date
    effective_to: date | None = None  # None = open-ended (still in force)


class EffectiveDatedSet:
    """The version history of one artefact family, e.g. `"expense_norms.statutory"`."""

    def __init__(self, family: str, versions: Sequence[EffectiveVersion]):
        self.family = family
        self.versions = tuple(sorted(versions, key=lambda v: v.effective_from))
        _check_no_gaps_or_overlaps(family, self.versions)

    def resolve(self, as_of: date) -> EffectiveVersion:
        """The version in force on `as_of`. Raises if none covers that date."""
        for v in self.versions:
            if v.effective_from <= as_of and (v.effective_to is None or as_of < v.effective_to):
                return v
        raise LookupError(f"no version of {self.family!r} is in force on {as_of}")

    def resolver_step(self, *, date_field: str = "decision_date", output: str | None = None):
        """A decider step: reads `date_field`, writes `output` (default `"<family>_version"`).

        Example::

            norms = EffectiveDatedSet("expense_norms.statutory", [...])
            flow(norms.resolver_step(), lookup_table, name="norms")
        """
        output = output or f"{self.family}_version"
        family = self.family
        versions = self.versions

        def resolve_version(decision_date: date) -> str:
            for v in versions:
                if v.effective_from <= decision_date and (v.effective_to is None or decision_date < v.effective_to):
                    return v.version_id
            raise LookupError(f"no version of {family!r} is in force on {decision_date}")

        resolve_version.__name__ = f"resolve_{family.replace('.', '_')}_version"
        s = step(resolve_version, output=output)
        if date_field != "decision_date":
            s = s.relabel(reads={"decision_date": date_field})
        return s


def _check_no_gaps_or_overlaps(family: str, versions: Sequence[EffectiveVersion]) -> None:
    for prev, nxt in zip(versions, versions[1:]):
        if prev.effective_to is None:
            raise ValueError(f"{family}: {prev.version_id} is open-ended but {nxt.version_id} follows it")
        if prev.effective_to != nxt.effective_from:
            raise ValueError(
                f"{family}: {prev.version_id} ends {prev.effective_to}, "
                f"{nxt.version_id} starts {nxt.effective_from} -- gap or overlap"
            )
