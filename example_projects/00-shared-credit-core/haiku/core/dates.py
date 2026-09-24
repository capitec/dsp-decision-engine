"""Effective-dated resolution of values (00 §6.2)."""
from __future__ import annotations
from datetime import date
from typing import Any, Optional


def resolve_effective_dated(
    value_by_date: dict[date, Any],
    as_at_date: date
) -> Optional[Any]:
    """
    Resolve a value effective on a given date.
    Returns the value with the latest effective-from date <= as_at_date.
    """
    candidates = {d: v for d, v in value_by_date.items() if d <= as_at_date}
    if not candidates:
        return None
    return candidates[max(candidates.keys())]
