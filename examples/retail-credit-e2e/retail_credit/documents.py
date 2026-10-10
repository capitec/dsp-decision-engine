"""The config documents `build` takes, and the shipped ones it falls back to when called without arguments."""
from __future__ import annotations

from pathlib import Path

from decikit import ConfigurableStep

SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "0.1.0"


def shipped(name: str) -> ConfigurableStep:
    return ConfigurableStep.load(str(SHIPPED / f"{name}.json"))


def recorded(stack: ConfigurableStep, prefix: str) -> ConfigurableStep:
    """The overlay stack, writing its set id and applied list as `<prefix>_adjustment_set_id` and `<prefix>_adjustments_applied`,
    because one decision applies overlays to several values."""
    return stack.relabel(writes={"adjustment_set_id": f"{prefix}_adjustment_set_id",
                                 "adjustments_applied": f"{prefix}_adjustments_applied"})
