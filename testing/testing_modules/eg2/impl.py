import polars as pl

def a(i1: pl.Series) -> pl.Series:
    return i1 + 1

def b(i2: pl.Series, a: pl.Series) -> pl.Series:
    return i2 + a

def c(b: pl.Series) -> pl.Series:
    return 2 * b