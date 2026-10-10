from __future__ import annotations

from typing import Iterator, Literal

from decikit import InterpretedEngine
from decikit.lab.checks._base import Check, CheckContext
from decikit.lab.checks._report import Finding, Severity
from decikit.lab.testing import assert_equivalent, corpus

__all__ = ["CrashProbe", "ModeDivergence"]


class CrashProbe(Check):
    """Runs the flow over the generated boundary cases (zero, negative, null, extremes) and reports crashes by case."""

    type: Literal["crash_probe"] = "crash_probe"
    code = "DCD501"
    executes = True
    severity: Severity = Severity.ERROR

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        exe = InterpretedEngine().bind(ctx.pipeline)
        for name, frame in corpus(ctx.pipeline).items():
            try:
                exe.run(frame, ctx.params)
            except Exception as e:
                yield ctx.finding(ctx.ir, f"case {name!r}: {type(e).__name__}: {e}",
                                  fix="guard the input the case moves, or declare it required", case=name)


class ModeDivergence(Check):
    """Every built-in engine (or `engines`) must agree on the generated cases (`testing.assert_equivalent`)."""

    type: Literal["mode_divergence"] = "mode_divergence"
    code = "DCD505"
    executes = True
    severity: Severity = Severity.ERROR
    engines: tuple[str, ...] | None = None

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for name, frame in corpus(ctx.pipeline).items():
            try:
                assert_equivalent(ctx.pipeline, frame, ctx.params, self.engines)
            except AssertionError as e:
                yield ctx.finding(ctx.ir, f"case {name!r}: {e}", fix="make the engines agree; see the message for the first difference",
                                  case=name)
            except Exception:
                pass  # a crash in the first mode is DCD501's finding
