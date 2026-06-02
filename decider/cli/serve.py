import sys
import click


_ENGINES = ("starlette", "sanic")


@click.command()
@click.option("--engine",  "-e", default="starlette", type=click.Choice(_ENGINES), show_default=True)
@click.option("--host",    "-h", default="0.0.0.0",   show_default=True)
@click.option("--port",    "-p", default=8080,         show_default=True, type=int)
@click.option("--workers", "-w", default=1,            show_default=True, type=int)
@click.option("--reload",        is_flag=True,         help="Enable hot-reload (dev only).")
def serve(engine: str, host: str, port: int, workers: int, reload: bool):
    """Start a Decider inference server.

    \b
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
        "decider.serving.servers.starlette:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        # factory=False so the module-level `app` is used directly
    )


def _serve_sanic(host: str, port: int, workers: int, reload: bool):
    try:
        from sanic import Sanic
    except ImportError:
        raise click.ClickException(
            "sanic is required for the sanic engine.\n"
            "Install it with:  pip install sanic"
        )
    click.echo(f"Starting Sanic server on {host}:{port}  workers={workers}")
    from decider.serving.servers.sanic import app
    app.run(
        host=host,
        port=port,
        workers=workers,
        auto_reload=reload,
        single_process=(workers == 1),
    )
