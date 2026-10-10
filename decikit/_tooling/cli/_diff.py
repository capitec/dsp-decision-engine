from __future__ import annotations

import json
from pathlib import Path

import click


@click.command()
@click.argument("base")
@click.argument("head")
@click.argument("entry")
@click.option("--data", help="A data file or a project dataset name; compares outputs over it, not just structure.")
@click.option("--key", help="A column that labels the rows.")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text", show_default=True)
def diff(base: str, head: str, entry: str, data: str | None, key: str | None, fmt: str) -> None:
    """Compare the pipeline ENTRY (module:factory) at git revisions BASE and HEAD.

    \b
    Without --data it compares structure: steps added, removed, renamed or
    changed, params and wiring. With --data it also runs both revisions over
    the data and reports which rows and columns change.
    Exit status is 0 when nothing differs, 1 when something does.
    """
    from decikit.lab import diff as api
    from decikit.exceptions import DecikitError

    revisions = api.revisions(base, head, entry=entry)
    try:
        outputs = revisions.outputs(_frame(data), key=key) if data else None
        structure = revisions.structure()
    except (DecikitError, ValueError) as e:
        raise click.ClickException(str(e)) from e
    differs = outputs is not None and bool(outputs.rows_changed or outputs.added_columns or outputs.removed_columns)
    differs = differs if outputs is not None else not structure.identical
    if fmt == "json":
        click.echo(json.dumps({"structure": structure.model_dump(mode="json"),
                               "outputs": outputs.model_dump(mode="json") if outputs else None}))
    else:
        click.echo(_text(structure, outputs))
    raise SystemExit(int(differs))


def _frame(data: str):
    from decikit.lab.data import dataset, read

    return read(data) if Path(data).exists() else dataset(data).load().frame


def _text(structure, outputs) -> str:
    lines = [f"structure: {'identical' if structure.identical else 'differs'}"]
    for label, refs in (("added", structure.added), ("removed", structure.removed)):
        lines += [f"  {label} {r.path}" for r in refs]
    lines += [f"  renamed {m.old.path} -> {m.new.path}" for m in structure.renamed]
    lines += [f"  moved {m.old.path} -> {m.new.path}" for m in structure.moved]
    for c in structure.changed_steps:
        lines.append(f"  changed {c.ref.path}: " + ", ".join(k for k in ("code", "consts", "inputs", "outputs", "params") if getattr(c, k)))
    lines += [f"  param {p.ref.path}.{p.name} {p.field}: {p.old} -> {p.new}" for p in structure.params]
    lines += [f"  edge {e.change} {e.edge.from_path} -> {e.edge.to_path}" for e in structure.edges]
    if outputs is not None:
        lines.append(f"outputs: {outputs.rows_changed} of {outputs.rows_compared} rows changed")
        for name, c in outputs.columns.items():
            lines.append(f"  {name}: {c.changed} changed" + (f", max abs {c.max_abs}" if c.max_abs is not None else ""))
        lines += [f"  added column {c}" for c in outputs.added_columns] + [f"  removed column {c}" for c in outputs.removed_columns]
    return "\n".join(lines)
