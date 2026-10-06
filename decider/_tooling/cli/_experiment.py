import click


@click.group()
def experiment() -> None:
    """Run experiments."""


@experiment.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", type=click.Path(file_okay=False), help="Write each scenario's output and result.json here.")
def run(path: str, out: str | None) -> None:
    """Run the experiment in PATH (an experiment.json); exit 1 unless every scenario completed.

    The status is one of completed, partial, non_reproducible or failed.
    """
    from decider.lab.experiments import RunStatus
    from decider.lab.experiments import run as run_experiment

    try:
        result = run_experiment(path, out_dir=out)
    except Exception as e:
        raise click.ClickException(f"{type(e).__name__}: {e}") from e
    click.echo(f"{result.name}: {result.status.value}")
    for sc in result.scenarios:
        click.echo(f"  {sc.name}: {sc.status.value}" + (f" ({sc.error})" if sc.error else ""))
    if out:
        click.echo(f"results in {out}/result.json")
    raise SystemExit(0 if result.status is RunStatus.COMPLETED else 1)
