"""Run a pipeline to the end in plain Python, so the editor's Python debugger stops at breakpoints in its steps.

    python -m decider._tooling.bridge.run app/flow.yaml --data rows.json
"""
from __future__ import annotations

import argparse
import sys

from decider import Engine
from decider._engine.run.runners import interpreted

from .bridge import Bridge
from .building import base_params

_CALLERS = (interpreted.InterpretedRunner._call.__code__, interpreted._frame.__code__)


def node_path() -> str | None:
    """The path of the flow node running on this thread, or `None` outside a run.

    An editor evaluates this at a breakpoint to tell which node a shared function was called for.

    Example::

        node_path()   # 'credit/affordable/eligible'
    """
    f = sys._getframe(1)
    while f is not None:
        if f.f_code in _CALLERS:
            return f.f_locals["node"].origin.path
        f = f.f_back
    return None


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m decider._tooling.bridge.run")
    p.add_argument("file", help="a pipeline .py, or a flow .yaml/.json")
    p.add_argument("--pipeline", help="the pipeline's name in a .py (default: the last one found)")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--data", help="rows: a JSON/CSV/Parquet file (default: SAMPLE or sample_request.json)")
    source.add_argument("--dataset", help="a dataset name from the project's datasets folder")
    p.add_argument("--row", type=int, help="run only this record: its 0-based position in the rows")
    a = p.parse_args(argv)
    b = Bridge()
    b.describe(a.file, a.pipeline)
    rows = b._rows(a.data, a.dataset, a.file)
    if a.row is not None:
        if not 0 <= a.row < rows.height:
            p.error(f"--row {a.row} is out of range: the data has {rows.height} rows")
        rows = rows.slice(a.row, 1)
    out = Engine().bind(b.step).run(rows, params=base_params(b.mod, None, b._build_params))
    print(out)


if __name__ == "__main__":
    main()
