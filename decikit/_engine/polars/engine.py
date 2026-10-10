from __future__ import annotations

from typing import Any

from decikit._engine.polars.host import PolarsExecutable
from decikit._engine.run.engine import Engine


class PolarsEngine:
    """Runs a pipeline as a polars `LazyFrame` plan: streamed, sunk or collected as you like.

    Every step still runs on the fused engine, batch by batch; a frame step
    sees the whole frame. `segments` is the engine that runs each batch.

    Example::

        exe = PolarsEngine().bind(pipeline)
        exe.run(pl.scan_parquet("in.parquet")).sink_parquet("out.parquet")
    """

    def __init__(self, segments: Engine | None = None):
        self.segments = Engine() if segments is None else segments

    def bind(self, step_or_ir: Any) -> PolarsExecutable:
        return PolarsExecutable(step_or_ir, lambda ir: self.segments.bind(ir, "fused"))
