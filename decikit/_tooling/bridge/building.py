"""Loading a pipeline the way `decikit serve` would, and the params it runs with."""
from __future__ import annotations

import json

from decikit import InterpretedEngine
from decikit.config import JsonFileStore
from decikit._engine.params import merge
from decikit.serving import RequestHandler
from decikit.serving._parse import coerce_record, has_date

from .loading import latest_config, project_root


def base_params(mod, params, default=None):
    """The module's `PARAMS` (its tables and tuned values, as the config store holds them), or `default`
    when it has none, with `params` on top."""
    return merge(getattr(mod, "PARAMS", None) or default or {}, params or {}) or None


def _date_inputs(step):
    return {i.name: i.annotation for i in InterpretedEngine().bind(step).plan.inputs if has_date(i.annotation)}


def build_step(fn, file):
    """Call a serving-style `def build(...)`: its config version's documents, like `decikit serve` would.

    Returns the built step, that version's `"params"` document (or `None`), and its
    `sample_request.json` record (or `None`, its dates converted like a request's), from the
    project root the pipeline's file sits in.
    """
    root = project_root(file) if file else None
    config = latest_config(root)
    step = RequestHandler(JsonFileStore(), fn).pipeline_fn(config)
    sample = None
    candidate = root and root / "sample_request.json"
    if candidate and candidate.exists():
        sample = coerce_record(json.loads(candidate.read_text()), _date_inputs(step))
    return step, config.get("params"), sample
