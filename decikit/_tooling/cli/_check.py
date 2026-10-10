from __future__ import annotations

import json

import click

from decikit._tooling.cli._diff import _frame


@click.command()
@click.argument("entry", required=False)
@click.option("--config", type=click.Path(dir_okay=False), help="A checks file; default is the project's checks.yaml.")
@click.option("--data", help="A data file or a project dataset name; enables the checks that need data.")
@click.option("--format", "fmt", type=click.Choice(["text", "json", "sarif"]), default="text", show_default=True)
@click.option("--fail-on", "fail_on", type=click.Choice(["error", "warning", "info"]),
              help="Exit 1 when a finding is at least this severe; default is the config's fail_on, else error.")
@click.option("--select", help="Keep only findings whose code starts with this prefix, e.g. DCD1.")
@click.option("--list", "list_", is_flag=True, help="List the registered checks instead of running them.")
def check(entry: str | None, config: str | None, data: str | None, fmt: str, fail_on: str | None,
          select: str | None, list_: bool) -> None:
    """Run the checks over the pipeline ENTRY (module:factory); exit 1 when the gate fails.

    \b
    Policy comes from the project's checks.yaml (or --config): which checks
    run, their severities and the default gate. Suppressed findings are
    shown but never fail the gate.
    """
    from decikit.lab import checks
    from decikit.exceptions import DecikitError
    from decikit.steps._entry import import_flow

    if list_:
        rows = checks.catalogue()
        if fmt == "json":
            click.echo(json.dumps(rows))
        else:
            for r in rows:
                flags = "".join(f" [{k}]" for k in ("needs_data", "executes") if r[k])
                click.echo(f"{r['code']}  {r['alias']}  {r['family']}  {r['severity']}{flags}")
        return
    if entry is None:
        raise click.UsageError("Missing argument 'ENTRY' (or pass --list).")
    try:
        report = checks.run(import_flow(entry), config=config, data=_frame(data) if data else None)
    except (DecikitError, ValueError) as e:
        raise click.ClickException(str(e)) from e
    if select:
        report = report.model_copy(update={
            "findings": tuple(f for f in report.findings if f.code.startswith(select)),
            "skipped": tuple(s for s in report.skipped if s.code.startswith(select))})
    click.echo({"text": report.to_text, "json": report.model_dump_json,
                "sarif": lambda: json.dumps(report.to_sarif())}[fmt]())
    raise SystemExit(int(report.fail_on(fail_on)))
