"""The config documents this package ships, for callers that do not pass their own."""
from __future__ import annotations

from pathlib import Path

from decider import ConfigurableStep

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "0.1.0"


def shipped(name: str) -> ConfigurableStep:
    """The step in `configs/0.1.0/<name>.json`.

    Example: `shipped("reason_codes").version`.
    """
    return ConfigurableStep.load(str(CONFIGS / f"{name}.json"))
