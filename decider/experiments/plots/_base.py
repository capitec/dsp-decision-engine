from __future__ import annotations

import json
import typing as t
import uuid

from pydantic import BaseModel, ConfigDict

from decider.exceptions import DeciderError
from decider.experiments.metrics._base import Summary
from decider.registry import BaseRegistryModule, ref


def require_altair():
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

        result.plots["divergence"].show()
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
        return {"text/html": self.html(), "text/plain": f"<{self.grammar} chart>"}

    def html(self) -> str:
        """The figure as an HTML fragment that draws itself, in a page or a notebook.

        Every figure on a page shares one load of Vega from the CDN; a chart
        whose scripts don't load says which URL failed.
        """
        div = f"vis-{uuid.uuid4().hex}"
        spec = json.dumps(self.spec).replace("</", "<\\/")
        return (f'<div id="{div}"></div><script>(function () {{\n'
                f'  const el = document.currentScript.previousElementSibling || document.getElementById("{div}");\n'
                f"  ({_loader()}).then(embed => embed(el, {spec}, {{mode: \"{self.grammar}\"}})).catch(err => {{\n"
                '    el.innerHTML = "<pre style=\\"color:#b91c1c;white-space:pre-wrap\\">" + err + "</pre>";\n'
                "  });\n})();</script>")


# One load of vega, vega-lite and vega-embed per page, however many figures it holds, in order and before any
# figure embeds: figures that each load their own race each other, and an empty vega-lite reads as
# "compile is not a function". Through requirejs when the page has it, as altair does; altair's own charts
# then see the libraries as loaded (VEGA_DEBUG) and don't load them again.
_LOADER = """window.deciderVega = window.deciderVega || new Promise(function (resolve, reject) {
  var urls = %(urls)s, names = ["vega", "vega-lite", "vega-embed"];
  function ready() {
    if (!window.vegaLite || typeof window.vegaLite.compile !== "function") {
      return reject(new Error("vega-lite did not load from " + urls[1] + "; is the CDN blocked?"));
    }
    window.VEGA_DEBUG = Object.assign(window.VEGA_DEBUG || {}, %(debug)s);
    resolve(window.vegaEmbed);
  }
  if (window.vegaEmbed && window.vegaLite && window.vega) return ready();
  if (typeof define === "function" && define.amd) {
    var paths = {};
    names.forEach(function (name, i) { paths[name] = urls[i] + "?noext"; });
    requirejs.config({paths: paths});
    return require(["vega-embed"], resolve, function (err) { reject(new Error("could not load " + err.requireModules)); });
  }
  urls.reduce(function (loaded, url) {
    return loaded.then(function () {
      return new Promise(function (done, fail) {
        var s = document.createElement("script");
        s.src = url;
        s.onload = done;
        s.onerror = function () { fail(new Error("could not load " + url + "; is the CDN blocked?")); };
        document.head.appendChild(s);
      });
    });
  }, Promise.resolve()).then(ready, reject);
})"""


def _loader() -> str:
    alt = require_altair()
    versions = (alt.VEGA_VERSION, alt.VEGALITE_VERSION, alt.VEGAEMBED_VERSION)
    urls = [f"https://cdn.jsdelivr.net/npm/{lib}@{v}" for lib, v in zip(("vega", "vega-lite", "vega-embed"), versions)]
    debug = dict(zip(("vega_version", "vegalite_version", "vegaembed_version"), versions))
    return _LOADER % {"urls": json.dumps(urls), "debug": json.dumps(debug)}


class Plot(BaseRegistryModule, root=True, group="plots"):
    """A pluggable plot of one metric's summaries, one per scenario.

    `metric` names the metric it draws (its `name`, or its type tag when
    unnamed), and keys the plot in `result.plots` unless `name` is set;
    `requires` is the `Summary` type it can draw, checked before the run
    starts; `width` and `height` size it in pixels. `render` returns an
    altair chart, or a full Vega spec for what
    Vega-Lite can't express; the result keeps it as a `Figure`.

    Example::

        import typing as t
        from decider.experiments.plots import Plot, require_altair

        class Totals(Plot):
            type: t.Literal["totals"] = "totals"
            requires: t.ClassVar = Total

            def render(self, summaries):
                alt = require_altair()
                rows = [{"scenario": name, "n": s.n} for name, s in summaries.items()]
                return alt.Chart(alt.Data(values=rows)).mark_bar().encode(x="scenario:N", y="n:Q")
    """

    requires: t.ClassVar[type[Summary]] = Summary
    metric: str
    name: str = ""
    width: int | None = None
    height: int | None = None

    @property
    def key(self) -> str:
        return self.name or self.metric

    def size(self) -> dict[str, int]:
        """The `width` and `height` set, in pixels, for `chart.properties(**self.size())`."""
        return {k: v for k, v in (("width", self.width), ("height", self.height)) if v is not None}

    def render(self, summaries: dict[str, Summary]) -> t.Any:
        raise NotImplementedError


PlotRef = ref(Plot)
"""A pydantic field type holding any registered `Plot`, chosen by its `type` tag."""
