"""Execution scope: run one selected record or the whole frame, with frame-step awareness."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from decider.contract import FlowDescription, RecordRef, StepRef
from decider.exceptions import DeciderError


class ExecutionScope(str, Enum):
    """What a run executes.

    `SELECTED_RECORD` runs one record, valid only when the flow has no frame
    step; `WHOLE_FRAME` runs the whole frame. Hybrid inspection — full-frame
    execution with one record's display, tracing and breakpoints focused — is
    `WHOLE_FRAME` with a `focus` record, not a third execution mode.
    """

    SELECTED_RECORD = "selected_record"
    WHOLE_FRAME = "whole_frame"


class ScopeError(DeciderError, ValueError):
    """A requested execution scope can't run this flow as asked."""

    _STATUS_CODE = 400


def frame_steps(description: FlowDescription) -> tuple[StepRef, ...]:
    """The frame steps of a flow: nodes that read and rewrite the whole frame, not one record."""
    return tuple(node for node in description.nodes if node.kind == "frame")


class ScopeCheck(BaseModel):
    """Whether a scope is valid for a flow, and the scope that will actually run.

    A `SELECTED_RECORD` scope over a flow with frame steps is redirected to
    `WHOLE_FRAME` (with `focus` kept for record-scoped observation); `redirected`
    is true and `reason` explains why.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    requested: ExecutionScope
    effective: ExecutionScope
    focus: RecordRef | None = None
    frame_steps: tuple[str, ...] = ()
    reason: str = ""

    @property
    def redirected(self) -> bool:
        return self.effective is not self.requested

    @property
    def hybrid(self) -> bool:
        return self.effective is ExecutionScope.WHOLE_FRAME and self.focus is not None


def check_scope(description: FlowDescription, scope: ExecutionScope, *,
                focus: RecordRef | None = None) -> ScopeCheck:
    """Validate `scope` for `description`; redirect a record-only scope over frame steps to whole-frame.

    Example::

        check = check_scope(describe(pipeline), ExecutionScope.SELECTED_RECORD, focus=ref)
        check.effective   # WHOLE_FRAME when the flow has a frame step, with check.redirected
    """
    if scope is ExecutionScope.SELECTED_RECORD:
        frames = frame_steps(description)
        if frames:
            return ScopeCheck(requested=scope, effective=ExecutionScope.WHOLE_FRAME, focus=focus,
                              frame_steps=tuple(f.path for f in frames),
                              reason="the flow has frame step(s) that read the whole frame; "
                                     "one record cannot supply them")
        return ScopeCheck(requested=scope, effective=scope, focus=focus)
    return ScopeCheck(requested=scope, effective=scope)
