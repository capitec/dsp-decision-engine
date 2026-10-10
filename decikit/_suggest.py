from __future__ import annotations

import difflib
from typing import Iterable

# A hint only decorates an error already being raised, so a loose match costs nothing. A caller
# that raises because of a match uses this stricter cutoff: a loose one would be a false error.
TYPO_CUTOFF = 0.8


def suggest(name: str, candidates: Iterable[str], cutoff: float = 0.6) -> str | None:
    # The candidate closest to `name`, never `name` itself.
    matches = difflib.get_close_matches(name, [c for c in candidates if c != name], n=1, cutoff=cutoff)
    return matches[0] if matches else None


def hint(name: str, candidates: Iterable[str], prefix: str = "") -> str:
    near = suggest(name, candidates)
    return f" Did you mean '{prefix}{near}'?" if near else ""
