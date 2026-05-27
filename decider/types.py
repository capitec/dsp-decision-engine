import typing as t
import polars as pl


TInputType = t.Dict[str, pl.LazyFrame]
TOutputType = pl.LazyFrame