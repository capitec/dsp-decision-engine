"""Scorecards: read variables and bins as tables and replace one variable's bins."""
from __future__ import annotations

import typing as t

from decider import ConfigurableStep
from decider._tooling.mcp._configs import Refusal, json_safe, read, refusing, save


def _load(path: str) -> dict:
    doc = read(path)
    if not isinstance(doc, dict) or doc.get("type") != "scorecard":
        raise Refusal(f"{path} is not a scorecard document")
    return doc


def _scored(variable: dict) -> dict:
    return variable["variable"] if variable["type"] == "adjusted" else variable


def _num(v: t.Any) -> str:
    return f"{{{v['param']}={v['default']}}}" if isinstance(v, dict) else str(v)


def _matches(b: dict) -> str:
    if "items" in b:
        return f"x in {b['items']}"
    lo, hi = b.get("lower_bound"), b.get("upper_bound")
    if lo is None and hi is None:
        return "x between its neighbours' bounds"
    return " and ".join(([f"x > {_num(lo)}"] if lo is not None else []) + ([f"x <= {_num(hi)}"] if hi is not None else []))


def _bins(variable: dict) -> str:
    rows = [f"| {i} | {_matches(b)} | {_num(b['value'])} | {b.get('name') or ''} |" for i, b in enumerate(variable["bins"])]
    d = variable["default"]
    rows.append(f"| default | no bin matches, or null | {_num(d['value'])} | {d.get('name') or ''} |")
    return "\n".join(["| # | matches | points | name |", "|---|---|---|---|", *rows])


def scorecard_view(path: str) -> dict:
    """A scorecard: each variable with its bins as a table (`{name=default}` marks a param), and the params it declares."""
    doc = _load(path)
    variables = []
    for v in doc["variables"]:
        if v["type"] == "constant":
            variables.append({"kind": "constant", "output": v.get("output_name", "constant_score"), "points": _num(v["score"])})
            continue
        s = _scored(v)
        points = s.get("value_output_name", "{variable_name}_score").format(variable_name=s["variable_name"])
        shown = {"variable": s["variable_name"], "kind": v["type"], "strict": s.get("strict", True), "output": points,
                 "bins_markdown": _bins(s), "bins": s["bins"], "default": s["default"]}
        if v["type"] == "adjusted":
            adjusted = v.get("variable_output_name", "{variable_name}_adjusted_score")
            shown |= {"scale": _num(v.get("scale", 1.0)), "offset": _num(v.get("offset", 0.0)), "points_output": points,
                      "output": adjusted.format(variable_name=s["variable_name"], score_value_output_name=points)}
        variables.append(shown)
    try:
        params = json_safe(dict(ConfigurableStep.load(doc).parameters()))
    except Exception as e:  # noqa: BLE001  a scorecard that does not build still shows its variables
        params = f"not available: {e}"
    return {"path": path, "name": doc["name"], "output": doc.get("output_name", "score"), "variables": variables,
            "params": params}


@refusing
def scorecard_set_bins(path: str, variable: str, bins: list[dict]) -> dict:
    """Replace the bins of `variable` (a `variable_name`); saved only if the scorecard still builds."""
    doc = _load(path)
    named = [_scored(v) for v in doc["variables"] if v["type"] != "constant"]
    for s in named:
        if s["variable_name"] == variable:
            s["bins"] = bins
            return save(path, doc)
    raise Refusal(f"no variable '{variable}'; variables are {[s['variable_name'] for s in named]}")
