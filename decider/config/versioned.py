import enum
import typing as t
from contextlib import contextmanager
from contextvars import ContextVar


class VersionPart(enum.Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"

class Version(t.NamedTuple):
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, version_str: str) -> "Version":
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version string: {version_str!r}")
        return cls(major=int(parts[0]), minor=int(parts[1]), patch=int(parts[2]))
    
    def bump(self, part: VersionPart) -> "Version":
        if part == VersionPart.MAJOR:
            return Version(self.major + 1, 0, 0)
        elif part == VersionPart.MINOR:
            return Version(self.major, self.minor + 1, 0)
        elif part == VersionPart.PATCH:
            return Version(self.major, self.minor, self.patch + 1)
        else:
            raise ValueError(f"Invalid version part: {part!r}")


    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    

class VersionedConfig(t.NamedTuple):
    version: Version
    config: t.Dict[str, t.Any]


_current_versioned_config: ContextVar[t.Optional[VersionedConfig]] = ContextVar("_current_versioned_config", default=None)

@contextmanager
def with_versioned_config(versioned_config: VersionedConfig) -> t.Iterator[VersionedConfig]:
    """Context manager to set the current versioned config for the duration of a block."""
    token = _current_versioned_config.set(versioned_config)
    try:
        yield versioned_config
    finally:
        _current_versioned_config.reset(token)

def get_current_versioned_config() -> t.Optional[VersionedConfig]:
    """Get the current versioned config from the context variable."""
    return _current_versioned_config.get()
