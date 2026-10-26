"""Plots: pluggable renderers of one metric's summaries, each requiring the `Summary` type it draws.

Plots draw with altair (the `experiments` extra), imported only when one renders.
"""
from decikit.lab.experiments.plots._bars import Bars, Changes
from decikit.lab.experiments.plots._base import Figure, Plot, PlotRef, require_altair
from decikit.lab.experiments.plots._distributions import Distribution, Ranges
from decikit.lab.experiments.plots._sankey import Sankey

__all__ = ["Bars", "Changes", "Distribution", "Figure", "Plot", "PlotRef", "Ranges", "Sankey", "require_altair"]
