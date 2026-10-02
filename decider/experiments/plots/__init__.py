"""Plots: pluggable renderers of one metric's summaries, each requiring the `Summary` type it draws.

Plots draw with altair (the `experiments` extra), imported only when one renders.
"""
from decider.experiments.plots.bars import Bars, Changes
from decider.experiments.plots.base import Figure, Plot, PlotRef
from decider.experiments.plots.sankey import Sankey

__all__ = ["Bars", "Changes", "Figure", "Plot", "PlotRef", "Sankey"]
