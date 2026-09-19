"""P07 scoring — 79 decision points. Spec 5.8. Elided from phases/__init__.py
"for length"; this package is the real file.

Scorecard SELECTION is 28 rules, not a lookup (segment x product x entry
point precedence is non-uniform — e.g. a joint application combines two
scorecards by the worse grade). Per-characteristic contributions are
REQUIRED OUTPUT, not a diagnostic: adverse-action wording is drawn directly
from the largest negative contributions, so this file's taps ARE the
explanation mechanism, not an add-on to it (doc 04 §4.1's "reason codes need
no new machinery").
"""

from __future__ import annotations

from decider2 import module, param, Branch

def scorecard_selection(segment_code: int, product_code: int, entry_point_code: int
                        ) -> str:
    """28 precedence rules. Segments 7, 8, 10 have no scorecard of their own
    and score on SC-A4 with a declared segment adjustment — a policy decision
    recorded as such, not a silent fallback."""
    pass  # resolve one of SC-A1..SC-S1/SC-P1 by the 28-rule precedence

def challenger_eligible(scorecard_selection: str, client_id: int,
                        challenger_traffic_share: float = param(0.12, ge=0, le=1)) -> bool:
    """12% of SC-A3 traffic also runs SC-A5. Declared off entirely on entry
    points 3 and 4 (entrypoints/manifest.py DECLARED_APPLICABILITY[3]):
    challenger share is defined over interactive volume."""
    pass  # deterministic sampling against challenger_traffic_share, keyed on client_id

def characteristic_contributions(scorecard_selection: str) -> list[dict]:
    """Per-characteristic, signed. Nulls are a SCORING BIN, not an error — the
    three null situations of spec 4.5 map to three different bins for 14 of
    SC-A3's 61 characteristics."""
    pass  # evaluate every characteristic's bin and signed point contribution

def score(characteristic_contributions: list[dict]) -> float:
    pass  # sum of characteristic_contributions' points

def challenger_score(challenger_eligible: bool, characteristic_contributions: list[dict]
                     ) -> float | None:
    """Recorded; never affects the client's decision."""
    pass  # evaluate SC-A5 in parallel where challenger_eligible; else None

Selection = module(scorecard_selection, challenger_eligible, characteristic_contributions,
                   score, challenger_score, name="selection",
                   taps=["characteristic_contributions"])
