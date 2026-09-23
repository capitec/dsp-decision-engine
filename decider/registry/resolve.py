"""Fuzzy name matching, for suggesting likely typos."""
from __future__ import annotations

import difflib
from typing import Iterable


_GATE_CUTOFF = 0.8
_HINT_CUTOFF = 0.6


def suggest_names(name: str, candidates: Iterable[str], *, n: int = 1) -> tuple[str, ...]:
    """Up to `n` closest candidates, closest first, for decorating an error
    that's already been raised. Generous cutoff: a false suggestion here
    costs nothing."""
    pool = [c for c in candidates if c != name]
    if not pool:
        return ()
    return tuple(difflib.get_close_matches(name, pool, n=n, cutoff=_HINT_CUTOFF))


def suggest_name(name: str, candidates: Iterable[str]) -> str | None:
    """The single closest candidate, for gating — i.e. when a non-None
    result is what makes the caller raise. Strict cutoff, because a false
    positive here is a false build error on legitimate code."""
    pool = [c for c in candidates if c != name]
    if not pool:
        return None
    matches = difflib.get_close_matches(name, pool, n=1, cutoff=_GATE_CUTOFF)
    return matches[0] if matches else None
