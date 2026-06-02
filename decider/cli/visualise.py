import os
import subprocess
import sys
from pathlib import Path

import click


@click.command()
@click.option(
    "--project-dir", "-d",
    default=None,
    metavar="DIR",
    help="Project directory (default: current directory).",
)
@click.option(
    "--ext-dir", "-e",
    default=None,
    metavar="DIR",
    help="Extension directory (default: <project-dir>/decider_extensions).",
)
@click.option(
    "--config-dir", "-c",
    default=None,
    metavar="DIR",
    help="Config directory (default: <project-dir>/configs).",
)
@click.option(
    "--root-module", "-r",
    default="main",
    show_default=True,
    metavar="KEY",
    help="Root module key inside the versioned config.",
)
@click.option(
    "--port", "-p",
    default=8501,
    show_default=True,
    type=int,
    help="Port for the Streamlit server.",
)
def visualise(
    project_dir: str,
    ext_dir: str,
    config_dir: str,
    root_module: str,
    port: int,
):
    """Launch the interactive module graph browser (Streamlit).

    \b
    Examples:
        decider visualise
        decider visualise --project-dir projects/loan_scoring
        decider visualise --root-module main --port 8502
    """
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "streamlit is required for this command.\n"
            "Install it with:  pip install streamlit"
        )

    app_file = Path(__file__).parent / "_visualise_app.py"
    resolved_project = Path(project_dir).resolve() if project_dir else Path.cwd()

    env = {
        **os.environ,
        "DECIDER_VISUALISE_PROJECT_DIR": str(resolved_project),
        "DECIDER_VISUALISE_ROOT_MODULE": root_module,
    }
    if ext_dir:
        env["DECIDER_VISUALISE_EXT_DIR"] = str(Path(ext_dir).resolve())
    if config_dir:
        env["DECIDER_VISUALISE_CONFIG_DIR"] = str(Path(config_dir).resolve())

    click.echo(f"Launching visualiser for {resolved_project.name} on http://localhost:{port}")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_file),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    try:
        result = subprocess.run(cmd, env=env)
        if result.returncode not in (0, -2):  # -2 = SIGINT
            sys.exit(result.returncode)
    except KeyboardInterrupt:
        pass
