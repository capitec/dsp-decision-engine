import typing as t
import polars as pl
from pydantic import BaseModel

class Eg3Config(BaseModel):
    multiplier: int = 0
    constant: float = 10.0


def a(i1: pl.Series, config: "Eg3Config") -> pl.Series:
    return i1 + config.constant

def b(i2: pl.Series, a: pl.Series, config: "Eg3Config") -> pl.Series:
    return i2 + a

def c(b: pl.Series, config: "Eg3Config") -> pl.Series:
    return config.multiplier * b