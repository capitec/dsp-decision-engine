# Numba types a namedtuple argument by its class object, and its on-disk cache
# pickles that class. A namedtuple made inside a function pickles by value and
# comes back as a different class in the next process, so every cache lookup
# would miss. Each class is therefore registered on this module under a name
# that encodes its fields, and `__getattr__` rebuilds any such name on demand.
from __future__ import annotations

import collections
import sys
import threading

_LOCK = threading.Lock()
_CLASSES: dict[tuple[str, ...], type] = {}
_PREFIX = "Bundle"


def _encode(fields: tuple[str, ...]) -> str:
    return _PREFIX + "".join(f"_{len(f)}_{f}" for f in fields)


def _decode(name: str) -> tuple[str, ...]:
    if not name.startswith(_PREFIX):
        raise ValueError(name)
    i, fields = len(_PREFIX), []
    while i < len(name):
        if name[i] != "_":
            raise ValueError(name)
        j = name.index("_", i + 1)
        n = int(name[i + 1:j])
        field = name[j + 1:j + 1 + n]
        if len(field) != n:
            raise ValueError(name)
        fields.append(field)
        i = j + 1 + n
    return tuple(fields)


def bundle_class(fields: tuple[str, ...]) -> type:
    """The namedtuple class for these field names; the same object on every call.

    Example::

        Params = bundle_class(("cap", "base_rate"))
        Params(48.0, 5.0).cap  # 48.0
    """
    fields = tuple(fields)
    cls = _CLASSES.get(fields)
    if cls is None:
        with _LOCK:
            cls = _CLASSES.get(fields)
            if cls is None:
                cls = collections.namedtuple("Params", fields, module=__name__)
                cls.__qualname__ = _encode(fields)
                setattr(sys.modules[__name__], cls.__qualname__, cls)
                _CLASSES[fields] = cls
    return cls


def __getattr__(name: str) -> type:
    try:
        fields = _decode(name)
    except ValueError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return bundle_class(fields)
