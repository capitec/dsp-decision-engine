"""Writing an experiment's results to a caller-selected directory.

`decider` owns the directory for the run, not long-term storage: it writes each
scenario's output as parquet, the run manifest and the computed summary and
Sankey as JSON, and returns the paths. The `experiments/<slug>/` project layout
is the caller's choice; a run result directory is generated under whatever
`out_dir` it names.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def write(out_dir: str | Path, result: Any, summary: dict[str, Any], sankey: dict[str, Any] | None,
          outputs: dict[str, dict | None]) -> dict[str, str]:
    """Write the run's outputs to `out_dir` and return `{name: path}` for the files written."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, output in outputs.items():
        if output is not None:
            path = root / f"{name}.parquet"
            pl.DataFrame(output).write_parquet(path)
            written[name] = str(path)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(result.manifest.model_dump(), indent=2, default=str))
    written["manifest"] = str(manifest_path)
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    written["summary"] = str(summary_path)
    if sankey is not None:
        sankey_path = root / "sankey.json"
        sankey_path.write_text(json.dumps(sankey, indent=2, default=str))
        written["sankey"] = str(sankey_path)
    return written
