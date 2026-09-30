from __future__ import annotations

from typing import Any, Literal

import polars as pl
from polars.testing import assert_frame_equal
from pydantic import BaseModel, ConfigDict

from decider.check import Check, Finding, Severity
from decider.checks._common import function_source
from decider.engine.ir.context import to_ir
from decider.engine.ir.nodes import CallNode, iter_nodes
from decider.testing import MODES, assert_equivalent, corpus


class Equivalence(BaseModel):
    """The result of comparing two revisions.

    `structurally_equivalent` is True only when both flows have identical
    structure and step source — a proof only for pure steps (non-determinism is
    flagged separately by `wall_clock`). Otherwise `method` is `"corpus"`: both
    revisions ran over generated threshold/regime frames and their outputs were
    compared; `equivalent` is True when none diverged, and `divergences` names
    the frames that did.

    Example::

        compare(term_old, term_new).equivalent
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    structurally_equivalent: bool
    method: Literal["structure", "corpus"]
    equivalent: bool
    divergences: tuple[str, ...] = ()
    frames_compared: int = 0


def equivalence(modes: tuple[str, ...] = MODES) -> Check:
    """Whole-path check: every execution mode must agree on the generated corpus.

    Reuses `testing.corpus` (the generated boundary/regime cases) and
    `testing.assert_equivalent` (cross-mode comparison), so this is the same
    evidence a test suite collects, surfaced as findings. A mode divergence —
    or a crash on any generated case — is an `error`-severity finding.

    Example::

        check.run(pipeline, suites=[checks.default_suite, (equivalence(),)])
    """

    def check(step: Any) -> list[Finding]:
        try:
            frames = corpus(step)
        except ValueError:
            return []  # the step reads no input; there are no cases to generate
        findings: list[Finding] = []
        for name, frame in frames.items():
            try:
                assert_equivalent(step, frame, modes=modes)
            except Exception as e:
                findings.append(Finding(check="equivalence", severity=Severity.ERROR,
                                        message=f"corpus {name!r}: {type(e).__name__}: {e}",
                                        detail={"frame": name}))
        return findings

    check.__name__ = "equivalence"
    return check


def compare(baseline: Any, revision: Any, frames: list[pl.DataFrame] | None = None) -> Equivalence:
    """Compare two revisions: structural equivalence where proven, corpus evidence otherwise.

    When both flows have identical structure and step source, the result is
    `method="structure"` (proven only for pure steps). Otherwise both revisions
    run over `frames` (default: the generated boundary corpus of `baseline`)
    and their outputs are compared row for row; `equivalent` is True only when
    nothing diverges.

    Example::

        result = compare(term_old, term_new)
        result.equivalent          # False
        result.divergences[0]      # "frame 0: ..."
    """
    if _signature(to_ir(baseline)) == _signature(to_ir(revision)):
        return Equivalence(structurally_equivalent=True, method="structure", equivalent=True)
    if frames is None:
        try:
            frames = [corpus(baseline)["boundary"]]
        except ValueError:
            frames = [pl.DataFrame()]
    divergences: list[str] = []
    for i, frame in enumerate(frames):
        try:
            assert_frame_equal(assert_equivalent(baseline, frame), assert_equivalent(revision, frame))
        except AssertionError as e:
            divergences.append(f"frame {i}: {e}")
    return Equivalence(structurally_equivalent=False, method="corpus",
                       equivalent=not divergences, divergences=tuple(divergences),
                       frames_compared=len(frames))


def _signature(root: Any) -> tuple[tuple, ...]:
    parts = []
    for node in iter_nodes(root):
        if isinstance(node, CallNode):
            src = function_source(node.fn)
            parts.append((node.origin.path, node.kind, node.origin.source, src[0] if src else None))
        else:
            parts.append((node.origin.path, type(node).__name__, node.origin.source))
    return tuple(parts)
