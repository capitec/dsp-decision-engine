"""The documents shipped in `configs/<version>/`, for a caller that has no config store."""
import json
from pathlib import Path

from decikit import ConfigurableStep

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def load(name: str, version: str = "0.1.0") -> ConfigurableStep:
    """A shipped step document, e.g. `load("reason_codes")`."""
    return ConfigurableStep.load(json.loads((CONFIGS / version / f"{name}.json").read_text()))
