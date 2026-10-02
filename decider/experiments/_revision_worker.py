"""Runs one experiment scenario against a checked-out git revision, in a fresh process.

Spawned by `runner._subprocess_run`. A revision's module shares the live
pipeline's import name, so running it in this process would either hand back
the already-imported live code (if only the entry module is evicted from
`sys.modules`) or risk it transitively, for anything the entry module itself
imports. A fresh process has no cache to collide with, at any depth. One JSON
request on stdin, one JSON reply on stdout — the envelope `decider.debug_bridge`
already uses for the same reason (an IDE-facing subprocess, not a library call).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

from decider.experiments.model import Scenario
from decider.experiments.runner import _fork_run, _fused_run, _load_flow_in


def main(stdin=sys.stdin, stdout=sys.stdout) -> None:
    try:
        request = json.loads(stdin.readline())
        step = _load_flow_in(request["entry"], Path(request["tree"]))
        frame = pl.read_parquet(request["frame_path"])
        sc = Scenario.model_validate({"name": "revision-worker", **request["scenario"]})
        run = _fused_run if request["mode"] == "fused" else _fork_run
        result = run(step, frame, request["params"], sc)
        reply = {"ok": True, "result": result}
    except Exception as e:  # noqa: BLE001  the parent process shows the message
        reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    stdout.write(json.dumps(reply, default=str) + "\n")
    stdout.flush()


if __name__ == "__main__":
    main()
