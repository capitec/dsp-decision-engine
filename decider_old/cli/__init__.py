import click

from .template import template
from .serve import serve
from .visualise import visualise


@click.group()
def cli():
    """Decider — build, serve and inspect decision pipelines."""


cli.add_command(template)
cli.add_command(serve)
cli.add_command(visualise)
