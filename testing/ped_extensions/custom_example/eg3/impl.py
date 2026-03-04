import typing as t
import polars as pl


def a(i1: pl.Expr, constant: float) -> pl.Expr:
    return i1 + constant

def b(i2: pl.Expr, a: pl.Expr) -> pl.Expr:
    return i2 + a

def c(b: pl.Expr, multiplier: int) -> pl.Expr:
    return multiplier * b