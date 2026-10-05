"""An experiment's report: a tree of blocks saying which summaries and charts to show, and where.

A `Report` lives in `experiment.json` next to the metrics and plots, so a UI
edits it the same way: each block is a registry model named by its `type`.
`result.show()` draws it in a notebook and `result.save_html(path)` writes it
as one page. Without one, the default report is an overview and a section
per metric. Subclass `Block` for your own.
"""
from decider.experiments.report._blocks import Block, BlockRef, Chart, MetricSection, Overview, Report, Row, Section, Table, Text

__all__ = ["Block", "BlockRef", "Chart", "MetricSection", "Overview", "Report", "Row", "Section", "Table", "Text"]
