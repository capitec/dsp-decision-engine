from __future__ import annotations

import difflib
from typing import Iterable

# Hints only decorate an error already being raised, so a loose match costs nothing.
_HINT_CUTOFF = 0.6
# A gate match makes the caller raise, so a loose match would be a false build error.
_GATE_CUTOFF = 0.8


def suggest_names(name: str, candidates: Iterable[str], *, n: int = 1) -> tuple[str, ...]:
    """Up to `n` candidates close to `name`, closest first, for did-you-mean hints.

    Example:
        suggest_names("tre", ["tree", "table"])  # ("tree",)
    """
    pool = [c for c in candidates if c != name]
    return tuple(difflib.get_close_matches(name, pool, n=n, cutoff=_HINT_CUTOFF))


def suggest_name(name: str, candidates: Iterable[str]) -> str | None:
    """The closest candidate to `name` under a strict cutoff, or `None`.

    For deciding whether a near-miss is likely a typo worth raising on.

    Example:
        suggest_name("incme", ["income", "age"])  # "income"
    """
    pool = [c for c in candidates if c != name]
    matches = difflib.get_close_matches(name, pool, n=1, cutoff=_GATE_CUTOFF)
    return matches[0] if matches else None
