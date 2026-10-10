import json
import os
import re
from pathlib import Path

import click
from dotenv import load_dotenv

from decikit import _settings as settings_module
from decikit._tooling.cli._check import check
from decikit._tooling.cli._diff import diff

__all__ = ["cli"]

TEMPLATES = Path(__file__).parent / "templates"
GUIDE = Path(__file__).parents[2] / "GUIDE.md"
CPU_TARGET_FILE = "decikit-cpu-target.json"


def _settings(**overrides) -> settings_module.DecikitSettings:
    # Through the environment so uvicorn/sanic worker processes read the same settings;
    # a ./.env fills in whatever the real environment doesn't set.
    load_dotenv(".env")
    for key, value in overrides.items():
        if value is not None:
            os.environ[f"DECIKIT_{key.upper()}"] = str(value)
    settings_module.get_settings.cache_clear()
    return settings_module.get_settings()


def _cpu_target_path(s: settings_module.DecikitSettings) -> Path:
    import numba

    # Next to numba's disk cache: NUMBA_CACHE_DIR if set, else the __pycache__ dirs under code_path.
    return Path(numba.config.CACHE_DIR or s.api.code_path, CPU_TARGET_FILE)


@click.group()
def cli() -> None:
    """decikit: build and serve decision pipelines.

    New here? Run `decikit guide` for the getting-started guide, then
    `decikit template NAME` for a starter project.

    Settings come from DECIKIT_* environment variables, or a .env file in the
    current directory, e.g. DECIKIT_API__PIPELINE=mypkg.pipeline:build,
    DECIKIT_CONFIG__BASEPATH=configs.
    """


@cli.command()
def guide() -> None:
    """Print the getting-started guide: concepts, examples, project layout, common mistakes."""
    click.echo(GUIDE.read_text())


@cli.command()
@click.argument("path", required=False, default=".", type=click.Path(path_type=Path))
@click.option("--fix", is_flag=True, help="On a duplicate id, assign a fresh id to the later copies instead of aborting.")
@click.option("--check", is_flag=True, help="Report what would change without writing anything.")
def ids(path: Path, fix: bool, check: bool) -> None:
    """Add durable `id=` tokens to every step and named flow missing one.

    \b
    Rewrites pipeline source in place, inserting the minimal `id="0123abcdef45"`
    fragment and changing no other byte. Refuses to run unless every affected
    file is git-tracked and the working tree is clean, and only writes after the
    whole tree parses and every id passes a repo-wide uniqueness scan.
    """
    from decikit._tooling.cli.ids import IdError, generate

    try:
        report = generate(path, fix=fix, check=check)
    except IdError as e:
        raise click.ClickException(f"{e}\n\nremediation: {e.remediation}") from e
    verb = "would add" if check else "added"
    click.echo(f"{verb} {len(report.changes)} id(s) in {report.files} file(s)")
    for f, lineno, name, id_ in report.changes:
        click.echo(f"  {f}:{lineno}  {name or '(anonymous)'}  ->  id={id_}")


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
    from decikit._engine.compile import cpu_target
    from decikit.serving._handler import construct_handler_from_settings

    s = _settings()
    try:
        staged = construct_handler_from_settings().stage(version)
    except Exception as e:
        raise click.ClickException(f"config version {version or 'latest'} failed to build: {type(e).__name__}: {e}") from e
    _cpu_target_path(s).write_text(json.dumps(cpu_target()))
    click.echo(f"built config version {staged} (pipeline {s.api.pipeline}, engine {s.api.engine.type})")


@cli.command()
@click.option("--host", help="Default: DECIKIT_SERVE__HOST or 0.0.0.0.")
@click.option("--port", type=int, help="Default: DECIKIT_SERVE__PORT or 8080.")
@click.option("--workers", type=int, help="Default: DECIKIT_SERVE__WORKERS or 2 * CPUs + 1.")
@click.option("--engine", help="An engine tag, e.g. interpreted, stepped or fused. Default: DECIKIT_API__ENGINE__TYPE or fused.")
@click.option("--server", type=click.Choice(["starlette", "sanic"]), help="Default: DECIKIT_SERVE__SERVER or starlette.")
def serve(host, port, workers, engine, server) -> None:
    """Serve the latest config version: POST /invocations, GET /ping.

    Each worker imports the pipeline and `inference.Handler` from
    DECIKIT_API__CODE_PATH (default: the current directory), then stages and
    activates the store's latest version before answering /ping. The
    engine's own settings come from DECIKIT_API__ENGINE__*, e.g.
    DECIKIT_API__ENGINE__STRICT_COMPILE=true.
    """
    from decikit._engine.compile import cpu_target

    s = _settings(serve__host=host, serve__port=port, serve__workers=workers, api__engine__type=engine,
                  serve__server=server)
    recorded = _cpu_target_path(s)
    built = tuple(json.loads(recorded.read_text())) if recorded.exists() else cpu_target()
    if built != cpu_target():
        # A warning, not a failure: startup warm-up still compiles everything before /ping answers.
        click.echo(
            f"warning: `decikit build` ran for CPU {built[1]!r} but this host is "
            f"{cpu_target()[1]!r}; numba's disk cache won't hit, so every worker compiles at startup. "
            "Run `decikit build` on this host type.",
            err=True,
        )
    workers = s.serve.workers or settings_module._default_workers()
    if s.serve.server == "sanic":
        _serve_sanic(s.serve.host, s.serve.port, workers)
        return
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException("serving with starlette needs uvicorn: pip install 'decikit[serve-starlette]'")
    uvicorn.run("decikit.serving._servers.starlette:create_app", factory=True,
                host=s.serve.host, port=s.serve.port, workers=workers)


def _serve_sanic(host: str, port: int, workers: int) -> None:
    try:
        from sanic import Sanic
        from sanic.worker.loader import AppLoader
    except ImportError:
        raise click.ClickException("serving with sanic needs sanic: pip install 'decikit[serve-sanic]'")
    from decikit.serving._servers.sanic import create_app

    # The loader calls the factory inside every worker process, so each one registers its own app.
    loader = AppLoader(factory=create_app)
    app = loader.load()
    app.prepare(host=host, port=port, workers=workers, single_process=workers == 1)
    if workers == 1:
        Sanic.serve_single(primary=app)
    else:
        Sanic.serve(primary=app, app_loader=loader)


@cli.command()
@click.option("--raw", is_flag=True, envvar="DECIKIT_MCP_RAW",
              help="Enable the raw record/trace/evaluate tools. Also set by DECIKIT_MCP_RAW=1.")
def mcp(raw: bool) -> None:
    """Serve the FastMCP server over stdio for an MCP host (e.g. an agent).

    \b
    Headless read tools (discovery, description, inspection, lineage, source
    context, run summaries, check reports, experiment definitions/results) are
    always available. Raw record/trace tools stay off unless `--raw` is set.
    Running a flow or experiment, starting the debugger and generating ids are
    marked destructive so the client confirms them before they run. The
    editor-bound selection, navigation and debug tools forward to the VS Code window
    that owns a workspace, over an authenticated per-window socket.
    """
    from decikit._tooling.mcp import run_stdio

    run_stdio(raw=raw)


from decikit._tooling.cli._experiment import experiment  # noqa: E402

cli.add_command(experiment)


@cli.command()
@click.argument("name")
@click.argument("directory", required=False, type=click.Path(file_okay=False, path_type=Path))
def template(name: str, directory: Path | None) -> None:
    """Write a starter project called NAME into DIRECTORY (default: ./NAME).

    The project directory is the package, so `import NAME` resolves to it and
    the repository root goes on sys.path.

    \b
    pipeline.py        steps and a build() function returning the pipeline
    inference.py       the Handler that serves it
    configs/0.0.0/     a config version with its params document
    sample_request.json one request, used by the tests and to warm up
    tests/             scores requests through the handler
    .env               DECIKIT_API__PIPELINE=NAME.pipeline:build and friends
    Then: cd NAME && pytest && decikit build && decikit serve
    """
    package = re.sub(r"\W+", "_", name).strip("_")
    if not package.isidentifier():
        raise click.ClickException(f"{name!r} can't name a Python package; start it with a letter, e.g. 'fraud_rules'")
    target = directory or Path(package)
    if target.exists() and any(target.iterdir()):
        raise click.ClickException(f"{target} exists and is not empty")
    for src in sorted(TEMPLATES.rglob("*")):
        if src.is_file() and "__pycache__" not in src.parts:
            rel = src.relative_to(TEMPLATES).as_posix().replace("{{name}}", package)
            # Stored as `env`: .gitignore, and so the wheel build, skips `.env`.
            dest = target / (".env" if rel == "env" else rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(src.read_text().replace("{{name}}", package))
            click.echo(f"created {dest}")


cli.add_command(check)
cli.add_command(diff)
