import polars as pl


def normalised(raw: pl.Expr) -> pl.Expr:
    """Z-score normalise the input series."""
    return (raw - raw.mean()) / raw.std()


def capped(normalised: pl.Expr) -> pl.Expr:
    """Cap the normalised series to [-3, 3]. Depends on `normalised`."""
    return normalised.clip(-3.0, 3.0)