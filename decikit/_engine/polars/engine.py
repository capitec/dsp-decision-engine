from __future__ import annotations

from typing import Any, Literal

from pydantic import model_validator

from decikit._engine.polars.host import PolarsExecutable
from decikit._engine.run.engines import Engine, FusedEngine


class PolarsEngine(Engine):
    """Runs a pipeline as a polars `LazyFrame` plan: collect it, stream it or sink it.

    Every step still runs on the fused engine, on each batch polars hands it;
    a frame step sees the whole frame, so the stream breaks there. A dict
    record runs on `segments` directly.

    Args:
        segments: the fused engine that runs every batch and every record;
            set `strict_compile` on it.
        params_validation: must equal `segments.params_validation`, which
            validates the document, so set both to change it.

    Example::

        exe = pipeline.compile("polars")
        exe.run(pl.scan_parquet("applications.parquet")).sink_parquet("decisions.parquet")
        PolarsEngine(segments=FusedEngine(strict_compile=True)).bind(pipeline)
    """

    type: Literal["polars"] = "polars"
    segments: FusedEngine = FusedEngine()

    @model_validator(mode="after")
    def _one_params_validation(self) -> PolarsEngine:
        if self.params_validation != self.segments.params_validation:
            raise ValueError(
                f"params_validation is {self.params_validation!r} on the polars engine but "
                f"{self.segments.params_validation!r} on its segments, which validate the document; set both "
                f"the same, e.g. PolarsEngine(params_validation='lazy', segments=FusedEngine(params_validation='lazy'))")
        return self

    def bind(self, step_or_ir: Any) -> PolarsExecutable:
        return PolarsExecutable(step_or_ir, self.segments.bind, self)
