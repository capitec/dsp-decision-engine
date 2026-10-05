from __future__ import annotations

import json
import typing as t
from html import escape

from pydantic import BaseModel, ConfigDict

from decider.exceptions import DeciderError
from decider.experiments.metrics._base import Metric, Summary
from decider.experiments.plots._base import Figure, Plot
from decider.registry import BaseRegistryModule, ref

if t.TYPE_CHECKING:
    from decider.experiments._experiments import ExperimentResult


class Block(BaseRegistryModule, root=True, group="blocks"):
    """A piece of a report: `html` renders it from a finished result.

    `metrics` and `plots` are the names it reads, which `validate` checks
    exist before the run.

    Example::

        class Note(Block):
            type: t.Literal["note"] = "note"
            text: str

            def html(self, result):
                return f"<aside>{escape(self.text)}</aside>"
    """

    def html(self, result: ExperimentResult) -> str:
        raise NotImplementedError

    def metrics(self) -> list[str]:
        return []

    def plots(self) -> list[str]:
        return []


BlockRef = ref(Block)
"""A pydantic field type holding any registered `Block`, chosen by its `type` tag."""


class Text(Block):
    """Markdown prose. Raw HTML in it is shown as text, not run.

    Example::

        Text(markdown="## Why\\nThe premium change only touches unsecured loans.")
    """

    type: t.Literal["text"] = "text"
    markdown: str

    def html(self, result: ExperimentResult) -> str:
        return f'<div class="text">{_markdown().render(self.markdown)}</div>'


class Overview(Block):
    """The run's status, size and revision, and a table of the scenarios and what each changed."""

    type: t.Literal["overview"] = "overview"

    def html(self, result: ExperimentResult) -> str:
        revision = result.manifest.revision
        records = result.manifest.input.row_count if result.manifest.input else None
        facts = [f"{records:,} records" if records is not None else "",
                 f"revision {(revision.resolved or revision.authored)[:12]}" if revision else "",
                 f"{len(result.scenarios) - 1} scenario(s) next to the baseline"]
        warning = ('<p class="warning">A rerun of the baseline differed: this flow is not reproducible.</p>'
                   if result.nondeterministic else "")
        rows = [{"scenario": s.name, "status": s.status.value, "records": s.records,
                 "revision": s.revision.resolved[:12] if s.revision and s.revision.resolved else "",
                 "changes": _changes(s), "error": s.error or ""} for s in result.scenarios]
        return (f'<h2>{escape(result.name)} <span class="status {result.status.value}">'
                f'{result.status.value}</span></h2><p class="facts">{escape(" · ".join(f for f in facts if f))}</p>'
                f"{warning}{_table(rows)}")


class Table(Block):
    """One metric's summaries as a table, a row per scenario.

    Example::

        Table(metric="divergence")
    """

    type: t.Literal["table"] = "table"
    metric: str

    def html(self, result: ExperimentResult) -> str:
        by_scenario = result.summaries.get(self.metric, {})
        return _table([{"scenario": name, **_flat(s.model_dump())} for name, s in by_scenario.items()])

    def metrics(self) -> list[str]:
        return [self.metric]


class Chart(Block):
    """One plot, by its name in `result.plots`.

    Example::

        Chart(plot="unsecured_rate")
    """

    type: t.Literal["chart"] = "chart"
    plot: str

    def html(self, result: ExperimentResult) -> str:
        return f'<div class="chart">{result.plots[self.plot].html()}</div>'

    def plots(self) -> list[str]:
        return [self.plot]


class MetricSection(Block):
    """A metric's heading, what it measures and where, its table, and every plot drawing it.

    A metric no plot draws gets the built-in plot for its summary type.

    Example::

        MetricSection(metric="payment")
    """

    type: t.Literal["metric_section"] = "metric_section"
    metric: str

    def html(self, result: ExperimentResult) -> str:
        definition = result.definition
        metric = next((m for m in definition.metrics if m.key == self.metric), None) if definition else None
        figures = [result.plots[p.key] for p in (definition.plots if definition else [])
                   if p.metric == self.metric and p.key in result.plots]
        by_scenario = result.summaries.get(self.metric, {})
        if not figures and by_scenario:
            figures = [Figure.of(p.render(by_scenario)) for p in _default_plots(self.metric, by_scenario)]
        charts = "".join(f'<div class="chart">{f.html()}</div>' for f in figures)
        return (f'<section class="metric"><h3>{escape(self.metric)}</h3><p class="where">{_where(metric)}</p>'
                f'{Table(metric=self.metric).html(result)}<div class="row">{charts}</div></section>')

    def metrics(self) -> list[str]:
        return [self.metric]


class Row(Block):
    """Blocks side by side, wrapping onto the next line when the page is narrow.

    Example::

        Row(blocks=[Chart(plot="rate_before"), Chart(plot="rate_after")])
    """

    type: t.Literal["row"] = "row"
    blocks: list[BlockRef]

    def html(self, result: ExperimentResult) -> str:
        return '<div class="row">' + "".join(f"<div>{b.html(result)}</div>" for b in self.blocks) + "</div>"

    def metrics(self) -> list[str]:
        return [m for b in self.blocks for m in b.metrics()]

    def plots(self) -> list[str]:
        return [p for b in self.blocks for p in b.plots()]


class Section(Block):
    """A titled group of blocks.

    Example::

        Section(title="Pricing", blocks=[Text(markdown="..."), Chart(plot="unsecured_rate")])
    """

    type: t.Literal["section"] = "section"
    title: str
    blocks: list[BlockRef]

    def html(self, result: ExperimentResult) -> str:
        return f"<section><h2>{escape(self.title)}</h2>" + "".join(b.html(result) for b in self.blocks) + "</section>"

    def metrics(self) -> list[str]:
        return [m for b in self.blocks for m in b.metrics()]

    def plots(self) -> list[str]:
        return [p for b in self.blocks for p in b.plots()]


class Report(BaseModel):
    """The report layout: a `title` and the blocks top to bottom.

    Example::

        Report(title="Premium sweep", blocks=[
            Overview(),
            Section(title="Rates", blocks=[Row(blocks=[Chart(plot="unsecured_rate"), Table(metric="divergence")])]),
        ])
    """

    model_config = ConfigDict(extra="forbid")

    title: str = ""
    blocks: list[BlockRef] = []


def default(result: ExperimentResult) -> Report:
    """An overview, then a section per metric."""
    return Report(blocks=[Overview(), *(MetricSection(metric=key) for key in result.summaries)])


def fragment(result: ExperimentResult) -> str:
    """The result's report (its own, or the default) as one HTML fragment, styles included."""
    report = (result.definition.report if result.definition else None) or default(result)
    title = f"<h1>{escape(report.title)}</h1>" if report.title else ""
    return f'<div class="decider-report"><style>{_CSS}</style>{title}' + "".join(
        b.html(result) for b in report.blocks) + "</div>"


def page(result: ExperimentResult) -> str:
    """The report as a standalone HTML page; its charts load Vega from the CDN when opened."""
    report = result.definition.report if result.definition else None
    title = escape((report.title if report and report.title else "") or result.name or "Experiment")
    return (f'<!DOCTYPE html>\n<html><head><meta charset="utf-8"><title>{title}</title></head>'
            f"<body>{fragment(result)}</body></html>\n")


def problems(report: Report, metrics: set[str], plots: set[str]) -> list[str]:
    """Report blocks that name a metric or plot the experiment doesn't have."""
    named = [(kind, name) for b in report.blocks for kind, names in (("metric", b.metrics()), ("plot", b.plots()))
             for name in names]
    return [f"the report shows {kind} {name!r}, which the experiment doesn't have"
            for kind, name in named if name not in (metrics if kind == "metric" else plots)]


def _markdown():
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        raise DeciderError("reports need markdown-it-py: pip install 'decider[experiments]'") from None
    return MarkdownIt("commonmark", {"html": False}).enable("table")


def _table(rows: list[dict[str, t.Any]]) -> str:
    if not rows:
        return '<p class="empty">nothing to show</p>'
    columns = list(dict.fromkeys(c for row in rows for c in row))
    head = "".join(f"<th>{escape(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{escape(_cell(row.get(c)))}</td>" for c in columns) + "</tr>"
                   for row in rows)
    return f'<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _cell(value: t.Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.6g}"
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else str(value)


def _changes(s) -> str:
    bits = []
    if s.params:
        bits.append(f"params {json.dumps(s.params)}")
    if s.overrides:
        where = (f" after {s.at}" if s.at else "") + (f" on record {s.row}" if s.row is not None else "")
        bits.append(f"set {json.dumps(s.overrides)}{where}")
    if s.revision:
        bits.append(f"revision {s.revision.authored}")
    return "; ".join(bits) or "unchanged"


def _where(metric: Metric | None) -> str:
    if metric is None:
        return ""
    column = getattr(metric, "column", None) or getattr(metric, "outcome", None)
    what = f"<code>{escape(metric.type)}</code>" + (f" of <code>{escape(column)}</code>" if column else "")
    if metric.node_id:
        return f"{what} at the step with id <code>{escape(metric.node_id)}</code>"
    if metric.nodes:
        return f"{what} at steps matching <code>{escape(metric.nodes)}</code>"
    return f"{what} in the flow's output"


def _default_plots(key: str, by_scenario: dict[str, Summary]) -> list[Plot]:
    summary = type(next(iter(by_scenario.values())))
    found = [cls for cls in _subclasses(Plot) if cls.requires is summary and cls.__module__.startswith("decider.")]
    return [found[0](metric=key)] if found else []


def _subclasses(cls: type) -> list[type]:
    return [c for sub in cls.__subclasses__() for c in (sub, *_subclasses(sub))]


def _flat(fields: dict[str, t.Any], prefix: str = "") -> dict[str, t.Any]:
    out: dict[str, t.Any] = {}
    for key, value in fields.items():
        if isinstance(value, dict):
            out.update(_flat(value, f"{prefix}{key}."))
        else:
            out[prefix + key] = _short(value) if isinstance(value, list) else value
    return out


def _short(values: list) -> str:
    if values and all(isinstance(v, (int, float)) for v in values):
        shown = ", ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in values[:6])
        return f"[{shown}{', …' if len(values) > 6 else ''}] ({len(values)})"
    text = json.dumps(values, default=str)
    return text if len(text) <= 80 else text[:77] + "..."


_CSS = """
.decider-report { font-family: system-ui, -apple-system, sans-serif; max-width: 1200px; line-height: 1.4; }
.decider-report h1 { font-size: 1.6em; margin: 0.4em 0; }
.decider-report h2 { font-size: 1.3em; margin: 1.2em 0 0.4em; }
.decider-report h3 { font-size: 1.1em; margin: 1em 0 0.2em; }
.decider-report .facts, .decider-report .where { color: #555; margin: 0.2em 0 0.6em; }
.decider-report .status { font-size: 0.7em; padding: 0.15em 0.6em; border-radius: 1em; vertical-align: middle;
  background: #fde68a; color: #713f12; }
.decider-report .status.completed { background: #bbf7d0; color: #14532d; }
.decider-report .status.failed { background: #fecaca; color: #7f1d1d; }
.decider-report .warning { background: #fef3c7; padding: 0.4em 0.8em; border-left: 4px solid #f59e0b; }
.decider-report .table { overflow-x: auto; margin: 0.4em 0; }
.decider-report table { border-collapse: collapse; font-size: 0.85em; }
.decider-report th, .decider-report td { border-bottom: 1px solid #e5e7eb; padding: 0.3em 0.7em; text-align: left;
  white-space: nowrap; }
.decider-report th { background: #f3f4f6; }
.decider-report .row { display: flex; flex-wrap: wrap; gap: 1.5em; align-items: flex-start; }
.decider-report section.metric { border-top: 1px solid #e5e7eb; margin-top: 1em; }
.decider-report code { background: #f3f4f6; padding: 0 0.3em; border-radius: 3px; }
"""
