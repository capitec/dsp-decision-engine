"""`decider2 serve <pipeline.py>` — doc 02 §3.6: "keep it thin."

This module's only job is finding a `Pipeline` and handing it to
`serving.app.app()` + `serving.server.run()`; everything that is actually
serving lives in `decider2/serving/`, and everything that is the params
lifecycle lives in `decider2/runtime/serve.py`. There is nothing here worth
unit-testing beyond "did it find the right object", so that is the only
thing `tests/test_cli.py` covers.

Usage:

    decider2 serve decider2/examples/flagship.py
    decider2 serve decider2/examples/flagship.py:pipeline   # explicit name
    decider2 serve decider2.examples.flagship               # a dotted module path also works
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import click

from decider2.graph.pipeline import Pipeline


def _import_module(path_part: str) -> ModuleType:
    path = Path(path_part)
    if path.suffix == ".py" or path.exists():
        if not path.exists():
            raise click.ClickException(f"no such file: {path}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise click.ClickException(f"could not import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    try:
        return importlib.import_module(path_part)
    except ImportError as exc:
        raise click.ClickException(f"could not import {path_part!r}: {exc}") from exc


def load_pipeline(target: str) -> Pipeline:
    """`target` is `path/to/file.py`, `path/to/file.py:name`, or a dotted
    module path (with an optional `:name` too). With no `:name`, a
    module-level `pipeline = ...` wins by convention; failing that, exactly
    one `Pipeline` instance anywhere in the module's namespace is accepted,
    and anything else is an error naming what to do about it."""
    path_part, _, attr = target.partition(":")
    module = _import_module(path_part)

    if attr:
        obj = getattr(module, attr, None)
        if not isinstance(obj, Pipeline):
            raise click.ClickException(f"{target}: {attr!r} is not a decider2 Pipeline")
        return obj

    conventional = getattr(module, "pipeline", None)
    if isinstance(conventional, Pipeline):
        return conventional

    candidates = [v for v in vars(module).values() if isinstance(v, Pipeline)]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise click.ClickException(
            f"{path_part}: no Pipeline found. Expose one as a module-level "
            f"`pipeline = ...`, or point at it explicitly with "
            f"`decider2 serve {path_part}:my_pipeline`."
        )
    raise click.ClickException(
        f"{path_part}: {len(candidates)} Pipelines found; point at one "
        f"explicitly with `decider2 serve {path_part}:name`."
    )


@click.group()
def cli() -> None:
    """decider2 — build, serve and inspect decision pipelines."""


@cli.command()
@click.argument("target")
@click.option(
    "--mode", type=click.Choice(["sealed", "live"]), default="sealed",
    help="doc 08 §4.1: 'live' additionally allows a change that would recompile.",
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, type=int, show_default=True)
def serve(target: str, mode: str, host: str, port: int) -> None:
    """Serve TARGET as a SageMaker-convention HTTP endpoint (doc 02 §3.6):
    GET /ping, POST /invocations, plus the params-play surface — GET/POST
    /params, GET /params/schema, POST /params/preview, POST /rollback — and
    GET /health.

    TARGET is a path to a .py file (optionally `path.py:name`) or a dotted
    module path exposing a decider2 `Pipeline`, conventionally as a
    module-level `pipeline = ...`.
    """
    from decider2.serving import app as build_app
    from decider2.serving.server import run

    pipeline = load_pipeline(target)
    asgi_app = build_app(pipeline, mode=mode)
    run(asgi_app, host=host, port=port)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
