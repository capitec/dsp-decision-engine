"""Build-time lints doc 03 promises but nothing previously implemented
(review finding 6).

`check_pipeline_file` (6a) is the one of the three that genuinely needs to
open and parse a whole FILE's source, unlike 6b/6c, which run for free at
`module()` composition from the objects already in hand:

    - 6a: a `def` in a pipeline file not present in the pipeline expression
      or a `module(...)` call (doc 03 §5.3: "Steps may not float") — this
      module.
    - 6b: a `params=` model field that no step reads (doc 03 §4) —
      `decider2.params.check_params_model_fields_are_read`, called from
      `decider2.graph.module.module()`.
    - 6c: a step parameter never referenced in the body — `decider2.params
      ._check_every_input_is_referenced`, called from
      `decider2.params.harvest_step`.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Iterable

from decider2.types import Module

__all__ = ["check_pipeline_file"]


def _load_module_object(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def _wired_functions(namespace: Iterable) -> set:
    """Every function object actually reachable as some step's `.fn`, from
    every `Module`/`Pipeline` instance in `namespace` — a `Pipeline` is
    walked via its `.elements` (a tuple of `Module`s) rather than importing
    `decider2.graph.pipeline.Pipeline` directly, so this module stays
    import-order-independent of the graph layer (duck-typed the same way
    `decider2.graph.module._apply` avoids a hard import at module scope)."""
    wired: set = set()
    for value in namespace:
        if isinstance(value, Module):
            wired.update(s.fn for s in value.steps)
        else:
            elements = getattr(value, "elements", None)
            if elements is None:
                continue
            for m in elements:
                if isinstance(m, Module):
                    wired.update(s.fn for s in m.steps)
    return wired


def check_pipeline_file(path: "str | Path") -> tuple[str, ...]:
    """Doc 03 §5.3: "Steps may not float." Every top-level `def` in a
    pipeline file must be reachable from a pipeline expression or a
    `module(...)` call somewhere in that same file, or it is silently dead
    — review finding 6a: an evaluation agent's central action-resolution
    step was dropped this way, and nothing said so.

    Returns the tuple of floating (unreachable) top-level function names,
    in file order; empty when there are none. Imports `path` to build the
    actual graph objects (needed to know what is truly wired — a name
    match in the AST alone can't tell a step from an ordinary helper), so
    this has the same side effects as running the file itself, exactly
    like `decider2.cli.load_pipeline` already does for `decider2 serve`.

    A top-level `def` that is a genuine, non-step helper (called from
    inside another step's body rather than composed directly) is flagged
    too — doc 03 §5.3's own shape is "one rule, one artefact, one `def`",
    so a `def` at pipeline-file scope is assumed to be step-shaped; a
    shared helper belongs in its own module, imported rather than defined
    alongside the pipeline.
    """
    path = Path(path)
    tree = ast.parse(path.read_text(), filename=str(path))
    top_level_defs = [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not top_level_defs:
        return ()

    mod = _load_module_object(path)
    wired = _wired_functions(vars(mod).values())

    return tuple(name for name in top_level_defs if vars(mod).get(name) not in wired)
