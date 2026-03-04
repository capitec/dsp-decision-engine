import polars as pl
from pydantic import BaseModel
from ped.modules.functional import generate_from_functions


class ExampleConfig(BaseModel):
    scale: float = 1.0
    offset: float = 0.0


# ── Functions using config ────────────────────────────────────────────────────

def scaled(x: pl.Expr, config: ExampleConfig) -> pl.Expr:
    """Scale the input series by config.scale."""
    return x * config.scale


def adjusted(scaled: pl.Expr, config: ExampleConfig) -> pl.Expr:
    """Shift the scaled value by config.offset. Depends on `scaled`."""
    return scaled + config.offset


# ── Functions without config ──────────────────────────────────────────────────

def clipped(x: pl.Expr) -> pl.Expr:
    """Clip values to the range [0, 1]."""
    return x.clip(0.0, 1.0)


def inverted(clipped: pl.Expr) -> pl.Expr:
    """Invert the clipped value: 1 - clipped. Depends on `clipped`."""
    return 1.0 - clipped


# ── Module definitions ────────────────────────────────────────────────────────

ModWithConfig = generate_from_functions("custom_example.eg1_with_config", scaled, adjusted)
ModWithoutConfig = generate_from_functions("custom_example.eg1_no_config", clipped, inverted)