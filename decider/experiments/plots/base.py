from __future__ import annotations

import typing as t
import uuid

from pydantic import BaseModel, ConfigDict

from decider.exceptions import DeciderError
from decider.experiments.metrics.base import Summary
from decider.registry import BaseRegistryModule, ref


def altair():
    """The `altair` module, or a `DeciderError` saying how to install it."""
    try:
        import altair
    except ImportError:
        raise DeciderError("experiment plots need altair: pip install 'decider[experiments]'") from None
    return altair


def ipython_display():
    """The `IPython.display` module, or a `DeciderError` saying how to install it."""
    try:
        import IPython.display
    except ImportError:
        raise DeciderError("showing results needs IPython: pip install 'decider[experiments]'") from None
    return IPython.display


class Figure(BaseModel):
    """A rendered plot: its JSON `spec` and the `grammar` it is written in.

    Shows itself in a notebook as the last value of a cell, or with `show()`.

    Example::

        result.plots["changes"].show()
    """

    model_config = ConfigDict(extra="forbid")

    grammar: t.Literal["vega-lite", "vega"]
    spec: dict[str, t.Any]

    @classmethod
    def of(cls, rendered: t.Any) -> Figure:
        """A `Figure` from what `Plot.render` returned: an altair chart, or a Vega or Vega-Lite spec."""
        if hasattr(rendered, "to_dict"):
            return cls(grammar="vega-lite", spec=rendered.to_dict())
        grammar = "vega-lite" if "vega-lite" in rendered.get("$schema", "") else "vega"
        return cls(grammar=grammar, spec=rendered)

    def show(self) -> None:
        ipython_display().display(self)

    def _repr_mimebundle_(self, include=None, exclude=None):
        alt = altair()
        if self.grammar == "vega-lite":
            return alt.Chart.from_dict(self.spec, validate=False)._repr_mimebundle_(include, exclude)
        from altair.utils.html import spec_to_html

        html = spec_to_html(self.spec, mode="vega", vega_version=alt.VEGA_VERSION,
                            vegaembed_version=alt.VEGAEMBED_VERSION, vegalite_version=alt.VEGALITE_VERSION,
                            output_div=f"vega-{uuid.uuid4().hex}", fullhtml=False)
        return {"text/html": html, "text/plain": f"<Vega chart: {self.spec.get('description', '')}>"}


class Plot(BaseRegistryModule, root=True):
    """A pluggable plot of one metric's summaries, one per scenario.

    `metric` names the metric it draws (its `name`, or its type tag when
    unnamed); `requires` is the `Summary` type it can draw, checked before the
    run starts. `render` returns an altair chart, or a full Vega spec for what
    Vega-Lite can't express; the result keeps it as a `Figure`.

    Example::

        class Totals(Plot):
            type: t.Literal["totals"] = "totals"
            requires: t.ClassVar = Total

            def render(self, summaries):
                alt = altair()
                rows = [{"scenario": name, "n": s.n} for name, s in summaries.items()]
                return alt.Chart(alt.Data(values=rows)).mark_bar().encode(x="scenario:N", y="n:Q")
    """

    requires: t.ClassVar[type[Summary]] = Summary
    metric: str
    name: str = ""

    @property
    def key(self) -> str:
        return self.name or self.type

    def render(self, summaries: dict[str, Summary]) -> t.Any:
        raise NotImplementedError


PlotRef = ref(Plot)
"""A pydantic field type holding any registered `Plot`, chosen by its `type` tag."""
