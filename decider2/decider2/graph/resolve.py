"""Name resolution: the did-you-mean for an unbound input (O23; doc 03 §2.2).

Wiring is by name (doc 03 §2), and an unbound name is, overwhelmingly, a
perfectly ordinary leaf/frame column — `net_income`, `expenses`. It is *not*
an error by default. It becomes one only when it is a near-miss of a name
that genuinely is in scope, because at that similarity a typo is far more
likely than two independently-chosen, coincidentally-similar column names
(doc 03 §2.2's worked example: `disposible_income` vs `disposable_income`).

Nothing here knows about `Module`, `Step` or a pipeline — it is a pure string
utility so `graph/interface.py` (module-local checks) and `graph/pipeline.py`
(cross-module checks) can both depend on it without a cycle between them.
"""
from __future__ import annotations

import difflib
from typing import Iterable

# Tuned, not measured: 'disposible_income' vs 'disposable_income' scores
# ~0.94 (one transposed/dropped vowel); unrelated real names in the flagship
# corpus (net_income, expenses, instalment, term_cap, min_net_salary) score
# well under this against each other. A future corpus of real typos (doc 00
# §2's `corpus` requirement) should replace this constant with a measured one.
_CUTOFF = 0.8


def suggest_name(name: str, candidates: Iterable[str]) -> str | None:
    """The single closest candidate to `name`, or `None` if nothing is close
    enough to be worth surfacing (doc 03 §2.2)."""
    pool = [c for c in candidates if c != name]
    if not pool:
        return None
    matches = difflib.get_close_matches(name, pool, n=1, cutoff=_CUTOFF)
    return matches[0] if matches else None
