"""Process-stable namedtuple bundle classes for `params`/`shared` (doc 03
§4.2) — the one place a bundle's Python class is ever created.

**Why a bundle class needs to be more than a memoised `namedtuple`.** A
step that takes a bare `params`/`shared` argument receives a namedtuple,
and numba types that argument as `types.NamedTuple`/`NamedUniTuple`, whose
identity (`.key`) is `(instance_class, member types)` — the *Python class
object itself*, not its field names. That is fine within one process once
the class is memoised (`bundle_class` below, keyed by prefix + fields, so
one field set is always one class). It is NOT fine across processes:
numba's on-disk cache index (`numba.core.caching.Cache._index_key`) is
`(argument types, magic_tuple, hashes)`, pickled with cloudpickle. A class
synthesised by `collections.namedtuple` inside a function is not an
attribute of its own `__module__`, so cloudpickle serialises it BY VALUE as
a dynamic class (with a fresh per-process uuid), and a later process
unpickles it into a *different* class object. Its `instance_class` then
never compares equal to the class that process built for the same fields,
every lookup misses, and every save appends a new entry — measured before
this module existed: a 3-band table's warm process re-saved all 14 of its
`shared`-taking specialisations (`_shared_get` x12, `row_fn`, `out_fn`) and
loaded only the 3 that take no bundle, with `_shared_get`'s index growing
by 12 entries per process, forever.

**The fix is to make the class picklable BY REFERENCE.** cloudpickle (and
stdlib pickle) pickle a class by `(__module__, __qualname__)` when
`getattr(sys.modules[__module__], __qualname__) is cls`. So every bundle
class is registered as an attribute of THIS module under a deterministic
qualname that is derived from — and fully encodes — its prefix and field
names (`_encode`/`_decode`), and the module's PEP 562 `__getattr__` re-
synthesises a class from that name on demand. Two consequences, both
required:

- **same fields -> same reference in every process.** The pickled index key
  is 75 bytes naming this module and the qualname, and unpickling it in a
  fresh process returns whatever `bundle_class()` returns there for the
  same fields, which is the class that process's `resolve_params` already
  handed the kernel. Key equal, genuine hit.
- **an index can name a field set this process has never built.** One
  helper dispatcher serves every table (`decider2.tables.encode`), so its
  index accumulates entries for every bundle class ever passed to it. A
  process that loads that index must resolve *all* of them or numba's
  `pickle.loads` raises `AttributeError` out of the compile — hence the
  name must be self-describing (a hash would not be invertible) and
  `__getattr__` must build, not look up.

Doc 05 §4.2's seven conditions are unaffected except condition 4 ("same
argument signature"), which this is what makes true for a bundle argument,
and condition 7 ("same `sys.modules` registration name"): the registration
name here is this module's own, fixed import name — never a hash, counter
or PID.

The class's `__name__` stays the readable prefix (`_shared_params`,
`_<step>_params`) so numba's type names and error messages read as before;
only `__qualname__` carries the encoded form.
"""
from __future__ import annotations

import collections
import sys
import threading

__all__ = ["bundle_class"]

_LOCK = threading.Lock()
_CLASSES: dict[tuple[str, tuple[str, ...]], type] = {}

# A qualname is `_QUAL_PREFIX` followed by one `_<length>_<text>` group per
# part, the prefix first, then each field in order. Every part is a valid
# Python identifier (`collections.namedtuple` enforces that for the type
# name and every field), so a length-prefixed encoding is unambiguous and
# the result contains only identifier characters — in particular never `.`
# (which pickle would split on) or `<locals>` (which it refuses).
_QUAL_PREFIX = "Bundle"


def _encode(prefix: str, fields: tuple[str, ...]) -> str:
    return _QUAL_PREFIX + "".join(f"_{len(part)}_{part}" for part in (prefix, *fields))


def _decode(qualname: str) -> tuple[str, tuple[str, ...]]:
    if not qualname.startswith(_QUAL_PREFIX):
        raise ValueError(qualname)
    i = len(_QUAL_PREFIX)
    parts: list[str] = []
    while i < len(qualname):
        if qualname[i] != "_":
            raise ValueError(qualname)
        j = qualname.index("_", i + 1)
        n = int(qualname[i + 1 : j])
        part = qualname[j + 1 : j + 1 + n]
        if len(part) != n:
            raise ValueError(qualname)
        parts.append(part)
        i = j + 1 + n
    if not parts:
        raise ValueError(qualname)
    return parts[0], tuple(parts[1:])


def bundle_class(prefix: str, fields: tuple[str, ...]) -> type:
    """The one namedtuple class for this `(prefix, fields)` — the same
    object every time in this process, and pickled by reference so that a
    numba cache entry keyed on it is found again by the next process."""
    fields = tuple(fields)
    key = (prefix, fields)
    cls = _CLASSES.get(key)
    if cls is not None:
        return cls
    with _LOCK:
        cls = _CLASSES.get(key)
        if cls is None:
            cls = collections.namedtuple(prefix, fields, module=__name__)
            qualname = _encode(prefix, fields)
            cls.__qualname__ = qualname
            setattr(sys.modules[__name__], qualname, cls)
            _CLASSES[key] = cls
    return cls


def __getattr__(name: str):
    """PEP 562: resolve a bundle class by its encoded qualname — what
    `pickle.loads` calls (through `getattr`) when a numba cache index
    written by another process names a class this one has not built
    yet."""
    try:
        prefix, fields = _decode(name)
    except ValueError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return bundle_class(prefix, fields)
