"""The config documents this example ships, for callers that embed the assessment without passing their own."""
from pathlib import Path

from decider import ConfigurableStep

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "0.1.0"


def shipped(name: str) -> ConfigurableStep:
    return ConfigurableStep.load(str(CONFIGS / f"{name}.json"))
