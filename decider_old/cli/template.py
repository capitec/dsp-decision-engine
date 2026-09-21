import os
import sys
from pathlib import Path

import click


@click.group()
def template():
    """Scaffold modules and projects from templates."""


@template.command("module")
@click.argument("class_name")
@click.option(
    "--package", "-p",
    default=None,
    metavar="PKG",
    help="Write into a uv src-layout package instead of a flat extension.",
)
@click.option(
    "--ext-dir", "-e",
    default=None,
    metavar="DIR",
    help="Extension directory (default: ./decider_extensions).",
)
def module_cmd(class_name: str, package: str, ext_dir: str):
    """Scaffold a new module called CLASS_NAME.

    \b
    Flat (inline):
        decider template module CreditScorer

    As a shareable package:
        decider template module CreditScorer --package mylib
    """
    from decider.templates.scaffold import write_inline_module, write_package_module

    ext_path = Path(ext_dir).resolve() if ext_dir else Path.cwd() / "decider_extensions"

    placeholder = (
        "def my_function(column_name: pl.Expr) -> pl.Expr:\n"
        "    return column_name * 1.0\n"
    )

    if package:
        mod_file, init_file = write_package_module(ext_path, class_name, package, placeholder)
        click.echo(click.style("created", fg="green") + f"  {mod_file}")
        click.echo(click.style("updated", fg="cyan")  + f"  {init_file}")
        pyproject = ext_path / package / "pyproject.toml"
        if pyproject.exists():
            click.echo(click.style("created", fg="green") + f"  {pyproject}")
    else:
        init_file = write_inline_module(ext_path, class_name, placeholder)
        click.echo(click.style("created", fg="green") + f"  {init_file}")

    click.echo(
        "\nEdit the file above, then load it with:\n"
        f"  from decider.initialization import initialize_decider\n"
        f"  initialize_decider(extension_path={str(ext_path)!r})"
    )


@template.command("project")
@click.argument("project_name")
@click.option(
    "--dir", "-d",
    "projects_dir",
    default=None,
    metavar="DIR",
    help="Parent directory for the new project (default: ./projects).",
)
def project_cmd(project_name: str, projects_dir: str):
    """Scaffold a new project directory called PROJECT_NAME."""
    from decider.templates.scaffold import write_project

    parent = Path(projects_dir).resolve() if projects_dir else Path.cwd() / "projects"
    try:
        project_dir = write_project(parent, project_name)
    except FileExistsError as e:
        raise click.ClickException(str(e))

    click.echo(click.style("created", fg="green") + f"  {project_dir}/")
    for f in sorted(project_dir.rglob("*")):
        if f.is_file():
            click.echo(f"  {f.relative_to(project_dir.parent)}")
