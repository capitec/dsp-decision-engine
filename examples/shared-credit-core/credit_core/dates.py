"""Effective-dated artefacts: pick the version in force on the record's own date, never on today's."""
from __future__ import annotations

from datetime import date
from typing import Any, Literal, NamedTuple, Sequence

from pydantic import BaseModel, ConfigDict, model_validator

from decider import ConfigurableStep, allow_fallback, step


class EffectiveVersion(NamedTuple):
    version_id: str
    effective_from: date
    effective_to: date | None = None  # None: still in force


# Being replaced by EffectiveVersions below; kept until the examples that import it are migrated.
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


class Version(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    version_id: str
    effective_from: date
    effective_to: date | None = None  # None: still in force


@allow_fallback
def resolve_version(decision_date: date, versions: tuple[tuple[str, int, int], ...], family: str) -> str:
    # Versions arrive as (id, first ordinal, ordinal it ends on or 0), because the diff cannot hash a date.
    on = decision_date.toordinal()
    for version_id, start, end in versions:
        if start <= on and (end == 0 or on < end):
            return version_id
    raise LookupError(f"no version of {family!r} is in force on {decision_date}")


class EffectiveVersions(ConfigurableStep):
    """The version of one artefact family in force on a record's own date, written to `output`.

    `name` defaults to `resolve_<family>_version` and `output` to `<family>_version`. Gaps and overlaps are rejected when
    the document loads.

    Example::

        norms = EffectiveVersions(family="norms", versions=[Version(version_id="v1", effective_from=date(2025, 1, 1),
                                                                    effective_to=date(2026, 1, 1)),
                                                            Version(version_id="v2", effective_from=date(2026, 1, 1))])
        norms.resolve(date(2025, 6, 1)).version_id   # "v1"
        flow(norms, lookup_table)                    # writes `norms_version` from `decision_date`
    """

    type: Literal["effective_versions"] = "effective_versions"
    family: str
    date_field: str = "decision_date"
    output: str = ""
    versions: tuple[Version, ...]

    @model_validator(mode="before")
    @classmethod
    def _defaults(cls, data: Any) -> Any:
        if isinstance(data, dict) and "family" in data:
            slug = data["family"].replace(".", "_")
            return {"name": f"resolve_{slug}_version", "output": f"{slug}_version", **data}
        return data

    @model_validator(mode="after")
    def _contiguous(self) -> EffectiveVersions:
        ordered = sorted(self.versions, key=lambda v: v.effective_from)
        for prev, nxt in zip(ordered, ordered[1:]):
            if prev.effective_to != nxt.effective_from:
                raise ValueError(f"{self.family}: {prev.version_id} ends {prev.effective_to}, "
                                 f"{nxt.version_id} starts {nxt.effective_from}: gap or overlap")
        return self

    def _packed(self) -> tuple[tuple[str, int, int], ...]:
        return tuple((v.version_id, v.effective_from.toordinal(), v.effective_to.toordinal() if v.effective_to else 0)
                     for v in sorted(self.versions, key=lambda v: v.effective_from))

    def resolve(self, as_of: date) -> Version:
        """The version in force on `as_of`; `LookupError` if none is."""
        version_id = resolve_version(as_of, self._packed(), self.family)
        return next(v for v in self.versions if v.version_id == version_id)

    def to_ir(self, ctx: Any) -> Any:
        return ctx.call(self, resolve_version, inputs={"decision_date": self.date_field},
                        outputs=[self.output], values={"versions": self._packed(), "family": self.family})
