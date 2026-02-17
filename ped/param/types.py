import typing as t

TVersionType = t.Any


class VersionedValue(t.NamedTuple):
    version: TVersionType
    value: t.Any

