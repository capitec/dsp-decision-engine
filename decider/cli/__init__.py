import json
import os
import re
from pathlib import Path

import click
from dotenv import load_dotenv

import decider.settings as settings_module

TEMPLATES = Path(__file__).parent / "templates"
GUIDE = Path(__file__).parent.parent / "GUIDE.md"
CPU_TARGET_FILE = "decider-cpu-target.json"


def _settings(**overrides) -> settings_module.DeciderSettings:
    # Through the environment so uvicorn/sanic worker processes read the same settings;
    # a ./.env fills in whatever the real environment doesn't set.
    load_dotenv(".env")
    for key, value in overrides.items():
        if value is not None:
            os.environ[f"DECIDER_{key.upper()}"] = str(value)
    settings_module.settings = settings_module.DeciderSettings()
    return settings_module.settings


def _cpu_target_path(s: settings_module.DeciderSettings) -> Path:
    import numba

    # Next to numba's disk cache: NUMBA_CACHE_DIR if set, else the __pycache__ dirs under code_path.
    return Path(numba.config.CACHE_DIR or s.api.code_path, CPU_TARGET_FILE)


@click.group()
def cli() -> None:
    """decider: build and serve decision pipelines.

    New here? Run `decider guide` for the getting-started guide, then
    `decider template NAME` for a starter project.

    Settings come from DECIDER_* environment variables, or a .env file in the
    current directory, e.g. DECIDER_API__PIPELINE=mypkg.pipeline:build,
    DECIDER_CONFIG__BASEPATH=configs.
    """


@cli.command()
def guide() -> None:
    """Print the getting-started guide: concepts, examples, project layout, common mistakes."""
    click.echo(GUIDE.read_text())


@cli.command()
@click.argument("version", required=False)
def build(version: str | None) -> None:
    """Stage config VERSION (default: the latest) exactly as `serve` would, then stop.

    \b
    Imports the pipeline, loads the version's documents, validates its params
    document, and runs a warm-up record through every kernel, which fills
    numba's disk cache for your step functions so `serve` starts without
    compiling them. Records the CPU target the cache was built for; `serve`
    warns when it runs on a different one.
    """
    from decider.engine.compile import cpu_target
    from decider.serving.handler import construct_handler_from_settings

    s = _settings()
    try:
        staged = construct_handler_from_settings().stage(version)
    except Exception as e:
        raise click.ClickException(f"config version {version or 'latest'} failed to build: {type(e).__name__}: {e}") from e
    _cpu_target_path(s).write_text(json.dumps(cpu_target()))
    click.echo(f"built config version {staged} (pipeline {s.api.pipeline}, mode {s.api.mode})")


@cli.command()
@click.option("--host", help="Default: DECIDER_SERVE__HOST or 0.0.0.0.")
@click.option("--port", type=int, help="Default: DECIDER_SERVE__PORT or 8080.")
@click.option("--workers", type=int, help="Default: DECIDER_SERVE__WORKERS or 2 * CPUs + 1.")
@click.option("--mode", type=click.Choice(["interpreted", "stepped", "fused"]), help="Default: DECIDER_API__MODE or fused.")
@click.option("--server", type=click.Choice(["starlette", "sanic"]), help="Default: DECIDER_SERVE__SERVER or starlette.")
def serve(host, port, workers, mode, server) -> None:
    """Serve the latest config version: POST /invocations, GET /ping.

    Each worker imports the pipeline and `inference.Handler` from
    DECIDER_API__CODE_PATH (default: the current directory), then stages and
    activates the store's latest version before answering /ping.
    """
    from decider.engine.compile import cpu_target

    s = _settings(serve__host=host, serve__port=port, serve__workers=workers, api__mode=mode, serve__server=server)
    recorded = _cpu_target_path(s)
    built = tuple(json.loads(recorded.read_text())) if recorded.exists() else cpu_target()
    if built != cpu_target():
        # A warning, not a failure: startup warm-up still compiles everything before /ping answers.
        click.echo(
            f"warning: `decider build` ran for CPU {built[1]!r} but this host is "
            f"{cpu_target()[1]!r}; numba's disk cache won't hit, so every worker compiles at startup. "
            "Run `decider build` on this host type.",
            err=True,
        )
    workers = s.serve.workers or settings_module._default_workers()
    if s.serve.server == "sanic":
        _serve_sanic(s.serve.host, s.serve.port, workers)
        return
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException("serving with starlette needs uvicorn: pip install 'decider[serve-starlette]'")
    uvicorn.run("decider.serving.servers.starlette:create_app", factory=True,
                host=s.serve.host, port=s.serve.port, workers=workers)


def _serve_sanic(host: str, port: int, workers: int) -> None:
    try:
        from sanic.worker.loader import AppLoader
    except ImportError:
        raise click.ClickException("serving with sanic needs sanic: pip install 'decider[serve-sanic]'")
    from decider.serving.servers.sanic import create_app

    # The loader calls the factory inside every worker process, so each one registers its own app.
    loader = AppLoader(factory=create_app)
    loader.load().run(host=host, port=port, workers=workers, single_process=workers == 1, app_loader=loader)


@cli.command()
@click.argument("name")
@click.argument("directory", required=False, type=click.Path(file_okay=False, path_type=Path))
def template(name: str, directory: Path | None) -> None:
    """Write a starter project called NAME into DIRECTORY (default: ./NAME).

    \b
    NAME/pipeline.py    steps and a build() function returning the pipeline
    NAME/inference.py   the Handler that serves it
    configs/0.0.0/      a config version with its params document
    sample_request.json one request, used by the tests and to warm up
    tests/              scores requests through the handler
    .env                DECIDER_API__PIPELINE=NAME.pipeline:build and friends
    Then: cd DIRECTORY && pytest && decider build && decider serve
    """
    target = directory or Path(name)
    if target.exists() and any(target.iterdir()):
        raise click.ClickException(f"{target} exists and is not empty")
    package = re.sub(r"\W+", "_", name).strip("_")
    if not package.isidentifier():
        raise click.ClickException(f"{name!r} can't name a Python package; start it with a letter, e.g. 'fraud_rules'")
    for src in sorted(TEMPLATES.rglob("*")):
        if src.is_file() and "__pycache__" not in src.parts:
            rel = src.relative_to(TEMPLATES).as_posix().replace("{{name}}", package)
            # Stored as `env`: .gitignore, and so the wheel build, skips `.env`.
            dest = target / (".env" if rel == "env" else rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(src.read_text().replace("{{name}}", package))
            click.echo(f"created {dest}")
