"""Backtest: run stored events through the same executable, with the overlay stack on and off."""
from __future__ import annotations

import polars as pl
from decikit import Executable

from fraud_interdiction.vocab import Action


def summarise(result: pl.DataFrame) -> dict:
    """Events, how many were held or stopped, the share that was, and the rand value stopped.

    Example::

        summarise(executable.run(events, params))   # {"events": 3, "blocked": 1, "hit_rate": 0.3333, "value_blocked": 9200.0}
    """
    blocked = result.filter(pl.col("action_code") >= Action.HOLD_FOR_REVIEW)
    return {"events": result.height, "blocked": blocked.height,
            "hit_rate": round(blocked.height / result.height, 4) if result.height else 0.0,
            "value_blocked": round(float(blocked["amount"].sum()), 2)}


def backtest(executable: Executable, events: pl.DataFrame, params: dict) -> dict:
    """`summarise` for the events as the overlays change them and as the authored rules alone would have."""
    without = {**params, "fraud_interdiction": {**params["fraud_interdiction"],
                                                "adjustment_stack_enabled": {"adjustment_stack_enabled": False}}}
    return {"with_overlays": summarise(executable.run(events, params)),
            "without_overlays": summarise(executable.run(events, without))}
