"""Runs one experiment scenario against a checked-out git revision, in a fresh process.

Spawned by the runner for a scenario on another revision: that revision's
module shares the live pipeline's import name, and only a process that never
imported the live one loads the checked-out code at every import depth. One
JSON request on stdin; each batch's `(output, node batches)` is pickled to
`<out>/<batch>.pkl`, and one JSON line on stdout says whether it worked.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import polars as pl

from decider.experiments._experiments import Scenario
from decider.steps._entry import import_flow
from decider.experiments._runner import _batches, _Local


def main(stdin=sys.stdin, stdout=sys.stdout) -> None:
    try:
        request = json.loads(stdin.read())
        step = import_flow(request["entry"], Path(request["tree"]))
        params = request["params"] if request["params"] is not None else step.parameters().defaults()
        local = _Local(step, params, Scenario(name="revision", **request["scenario"]), request["selectors"])
        out = Path(request["out"])
        out.mkdir(parents=True, exist_ok=True)
        for index, (offset, batch) in enumerate(_batches(pl.scan_parquet(request["input"]), request["batch_size"])):
            (out / f"{index}.pkl").write_bytes(pickle.dumps(local.run(index, offset, batch)))
        reply = {"ok": True}
    except Exception as e:  # noqa: BLE001  the parent process shows the message
        reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    stdout.write(json.dumps(reply) + "\n")
    stdout.flush()


if __name__ == "__main__":
    main()
