import typing as t
import click


_ENGINES = ("starlette", "sanic")


def _settings():
    from decider.settings import settings
    return settings.serve


def _default_host() -> str:
    return _settings().host


def _default_port() -> int:
    return _settings().port


def _default_workers() -> int:
    from decider.settings import _default_workers as _dw
    return _settings().workers or _dw()


@click.command()
@click.option("--engine",  "-e", default="starlette",      type=click.Choice(_ENGINES), show_default=True)
@click.option("--host",    "-h", default=_default_host,    show_default=True)
@click.option("--port",    "-p", default=_default_port,    show_default=True, type=int)
@click.option("--workers", "-w", default=_default_workers, show_default=True, type=int)
@click.option("--reload",        is_flag=True,             help="Enable hot-reload (dev only).")
def serve(engine: str, host: str, port: int, workers: int, reload: bool):
    """Start a Decider inference server.

    \b
    host, port and workers read from DECIDER_SERVE__HOST / _PORT / _WORKERS
    when not supplied on the command line.  workers falls back to nproc*2+1.

    Examples:
        decider serve
        decider serve --engine sanic --workers 4
        decider serve --port 9000 --reload
    """
    if engine == "starlette":
        _serve_starlette(host, port, workers, reload)
    else:
        _serve_sanic(host, port, workers, reload)


def _serve_starlette(host: str, port: int, workers: int, reload: bool):
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "uvicorn is required for the starlette engine.\n"
            "Install it with:  pip install uvicorn"
        )
    click.echo(f"Starting Starlette server on {host}:{port}  workers={workers}")
    uvicorn.run(
        "decider.serving.servers.starlette:get_app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        factory=True,
    )


def _serve_sanic(host: str, port: int, workers: int, reload: bool):
    try:
        from sanic import Sanic
        from sanic.worker.loader import AppLoader
    except ImportError:
        raise click.ClickException(
            "sanic is required for the sanic engine.\n"
            "Install it with:  pip install sanic"
        )
    click.echo(f"Starting Sanic server on {host}:{port}  workers={workers}")
    from decider.serving.servers.sanic import create_app
    # AppLoader ensures the factory is called inside each worker process so
    # Sanic's _app_registry is populated correctly in every worker — not just
    # the main process.
    loader = AppLoader(factory=create_app)
    app = loader.load()
    app.run(
        host=host,
        port=port,
        workers=workers,
        auto_reload=reload,
        single_process=(workers == 1),
        app_loader=loader,
    )
