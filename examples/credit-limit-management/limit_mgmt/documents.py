"""The config documents this example ships, for callers of `pipeline.build` that pass none of their own."""
from __future__ import annotations

import json
from pathlib import Path

from decider import ConfigurableStep

SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "0.1.0"


def shipped(name: str) -> ConfigurableStep:
    """`configs/0.1.0/<name>.json` loaded as the step it describes."""
    return ConfigurableStep.load(json.loads((SHIPPED / f"{name}.json").read_text()))
