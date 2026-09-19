"""Banding. Record tier.

Band edges are an artefact with declared closure, not an `if` chain. s5.4 makes
this a requirement rather than a preference: "band edges are closed at the lower
bound and open at the upper, without exception" is a property of the artefact,
and an auditor can read it without reading Python.
"""

from decider2 import bands, module, param, table

UtilisationBands = bands(
    "clm.utilisation_bands",
    edges=[0.0, 0.0001, 0.10, 0.25, 0.40, 0.55, 0.70, 0.90],
    lower_closed=True,          # 40.0% falls in the 40-55% band. Declared, not discovered.
    labels=["zero", "0-10", "10-25", "25-40", "40-55", "55-70", "70-90", "90+"],
    effective_dated=True,       # scenario 5: 8 bands becomes 9, historically resolvable
)

MobBands = bands(
    "clm.mob_bands",
    edges=[6, 12, 18, 24, 36, 60],
    lower_closed=True,
    labels=["6-11", "12-17", "18-23", "24-35", "36-59", "60+"],
    effective_dated=True,
)


def utilisation_band(mean_utilisation_6m: float, tables) -> int:
    """The matrix keys off the 6-month mean, never the spot value (s5.1)."""
    return tables.utilisation_bands.index(mean_utilisation_6m)


def mob_band(months_on_book: int, tables) -> int:
    """Months-on-book band at decision_date."""
    return tables.mob_bands.index(months_on_book)


def band_edge_flag(
    mean_utilisation_6m: float,
    months_on_book: int,
    tables,
    tolerance: float = param(0.01, gt=0, le=0.10,
                             description="fraction of band width counted as an edge"),
) -> bool:
    """True when any banded input sits within `tolerance` of a boundary (s5.4)."""
    pass  # OR over the three band tables' `.near_edge(value, tolerance)`


Banding = module(utilisation_band, mob_band, band_edge_flag, name="banding",
                 evidence=["utilisation_band", "mob_band", "band_edge_flag"])
