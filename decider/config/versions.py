from __future__ import annotations

import enum
import typing as t


class VersionPart(enum.Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


class Version(t.NamedTuple):
    """A `major.minor.patch` config version; compares and sorts numerically."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, version: str | Version) -> Version:
        """`Version.parse("1.2.0") == Version(1, 2, 0)`; a `Version` passes through."""
        if isinstance(version, Version):
            return version
        parts = version.split(".")
        if len(parts) != 3:
            raise ValueError(f"invalid version {version!r}; expected 'major.minor.patch'")
        return cls(*(int(p) for p in parts))

    def bump(self, part: VersionPart) -> Version:
        if part is VersionPart.MAJOR:
            return Version(self.major + 1, 0, 0)
        if part is VersionPart.MINOR:
            return Version(self.major, self.minor + 1, 0)
        return Version(self.major, self.minor, self.patch + 1)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


class VersionedConfig(t.NamedTuple):
    """One stored version: `config` maps a key (`"params"`, `"tree"`) to its document."""

    version: Version
    config: dict[str, t.Any]
