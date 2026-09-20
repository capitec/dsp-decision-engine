"""Name resolution: the one did-you-mean implementation in the package (O23;
doc 03 §2.2).

Wiring is by name (doc 03 §2), and an unbound name is, overwhelmingly, a
perfectly ordinary leaf/frame column — `net_income`, `expenses`. It is *not*
an error by default. It becomes one only when it is a near-miss of a name
that genuinely is in scope, because at that similarity a typo is far more
likely than two independently-chosen, coincidentally-similar names (doc 03
§2.2's worked example: `disposible_income` vs `disposable_income`).

This used to be two differently-tuned implementations — `graph/resolve.py`'s
single-suggestion helper (cutoff 0.8) and `runtime/invoke.py`'s inline
`_unknown_namespace_error` (cutoff 0.6, up to 3 suggestions) — found during an
over-engineering audit and unified here: one helper, one cutoff, used by
both. It lives at the package root rather than under `graph/` on purpose:
`decider2.runtime.invoke` must not import `decider2.graph` (its own docstring
— "keeps the dependency direction one-way and this layer buildable/testable
before or without the graph layer"), so a helper shared by both has to sit
somewhere neither layer's absence would break. Nothing here knows about
`Module`, `Step` or a pipeline — it is a pure string utility over stdlib
`difflib` only.
"""
from __future__ import annotations

import difflib
from typing import Iterable

# TWO cutoffs, because the suggestion plays two different roles and a false
# positive costs wildly different amounts in each.
#
# GATE: the suggestion DECIDES whether to raise (graph/interface.py,
# graph/pipeline.py's unbound-input check) — an unproduced name is either a
# legitimate leaf from the frame or a typo, and "is it close to something we
# produce?" is the only signal available. A false positive here is a FALSE
# BUILD ERROR that blocks correct code, so it must be strict.
#
# HINT: the error is already decided and the suggestion only improves the
# message (runtime/invoke.py's unknown namespace, pipeline.emit's unknown
# name). A false positive here costs nothing but a slightly odd sentence, so
# it can be generous and catch more real typos.
#
# Measured against a credit vocabulary rather than the flagship's five
# deliberately-distinct names: at 0.6, **8 of 10** legitimate domain pairs
# collide (net_income~gross_income, balance~balance_cents, score~score_band,
# rate~rate_card...). At 0.8, 2 of 10 — and one of those, term_cap~term_cap_a,
# is already handled by excluding a step's own output from its pool. A real
# typo like "kap"->"cap" scores 0.667, so it is caught by HINT and not by
# GATE; that is the right trade, because "kap" as an unbound name still
# errors, it just errors without a suggestion.
_GATE_CUTOFF = 0.8
_HINT_CUTOFF = 0.6


def suggest_names(name: str, candidates: Iterable[str], *, n: int = 1) -> tuple[str, ...]:
    """Up to `n` closest candidates, closest first, for DECORATING an error
    that has already been decided (doc 03 §2.2). Generous: a false suggestion
    costs nothing here."""
    pool = [c for c in candidates if c != name]
    if not pool:
        return ()
    return tuple(difflib.get_close_matches(name, pool, n=n, cutoff=_HINT_CUTOFF))


def suggest_name(name: str, candidates: Iterable[str]) -> str | None:
    """The single closest candidate for GATING — i.e. when a non-None result
    is what makes the caller raise (doc 03 §2.2). Strict, because a false
    positive is a false build error on legitimate code."""
    pool = [c for c in candidates if c != name]
    if not pool:
        return None
    matches = difflib.get_close_matches(name, pool, n=1, cutoff=_GATE_CUTOFF)
    return matches[0] if matches else None
