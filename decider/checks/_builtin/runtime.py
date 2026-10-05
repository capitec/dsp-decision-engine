from __future__ import annotations

from typing import Iterator, Literal

from decider import Engine
from decider.checks._base import Check, CheckContext
from decider.checks._report import Finding, Severity
from decider.testing import MODES, assert_equivalent, corpus

__all__ = ["CrashProbe", "ModeDivergence"]


class CrashProbe(Check):
    """Runs the flow over the generated boundary cases (zero, negative, null, extremes) and reports crashes by case."""

    type: Literal["crash_probe"] = "crash_probe"
    code = "DCD501"
    executes = True
    severity: Severity = Severity.ERROR

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        exe = Engine().bind(ctx.pipeline)
        for name, frame in corpus(ctx.pipeline).items():
            try:
                exe.run(frame, ctx.params)
            except Exception as e:
                yield ctx.finding(ctx.ir, f"case {name!r}: {type(e).__name__}: {e}",
                                  fix="guard the input the case moves, or declare it required", case=name)


class ModeDivergence(Check):
    """Interpreted, stepped and fused modes must agree on the generated cases (`testing.assert_equivalent`)."""

    type: Literal["mode_divergence"] = "mode_divergence"
    code = "DCD505"
    executes = True
    severity: Severity = Severity.ERROR
    modes: tuple[str, ...] = MODES

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for name, frame in corpus(ctx.pipeline).items():
            try:
                assert_equivalent(ctx.pipeline, frame, ctx.params, self.modes)
            except AssertionError as e:
                yield ctx.finding(ctx.ir, f"case {name!r}: {e}", fix="make the modes agree; see the message for the first difference",
                                  case=name)
            except Exception:
                pass  # a crash in the first mode is DCD501's finding
