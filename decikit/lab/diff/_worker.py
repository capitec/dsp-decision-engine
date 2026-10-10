"""Loads one revision's flow in a fresh process: its snapshot and, given an input, its output.

One JSON request on stdin; results are written under the request's `out`
directory, and one JSON line on stdout says whether it worked.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import polars as pl

from decikit.lab.diff._hash import snapshot
from decikit.steps._entry import import_flow


def main(stdin=sys.stdin, stdout=sys.stdout) -> None:
    try:
        request = json.loads(stdin.read())
        step = import_flow(request["entry"], Path(request["tree"]))
        out = Path(request["out"])
        (out / "snapshot.pkl").write_bytes(pickle.dumps(snapshot(step)))
        if request.get("input"):
            step.compile("interpreted").run(pl.read_parquet(request["input"]), request["params"]).write_parquet(
                out / "output.parquet")
        reply = {"ok": True}
    except Exception as e:  # noqa: BLE001  the parent process shows the message
        reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    stdout.write(json.dumps(reply) + "\n")
    stdout.flush()


if __name__ == "__main__":
    main()
