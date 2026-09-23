from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

# Each `kind` names the `Session` method it calls, so `Session.apply` dispatches by name.


@dataclass(frozen=True, slots=True)
class BreakAt:
    """Break at a path, a path prefix, or `path#locator` inside a row node."""

    target: str
    kind: Literal["break_at"] = "break_at"


@dataclass(frozen=True, slots=True)
class ClearBreak:
    """Clear a breakpoint set with the same `target`."""

    target: str
    kind: Literal["clear_break"] = "clear_break"


@dataclass(frozen=True, slots=True)
class StepOver:
    """Advance one node, running a branch or loop to its end."""

    kind: Literal["step"] = "step"


@dataclass(frozen=True, slots=True)
class StepInto:
    """Advance to the very next checkpoint."""

    kind: Literal["step_into"] = "step_into"


@dataclass(frozen=True, slots=True)
class Resume:
    """Run to the next breakpoint or the end."""

    kind: Literal["resume"] = "resume"


@dataclass(frozen=True, slots=True)
class Pause:
    """Stop at the next checkpoint."""

    kind: Literal["pause"] = "pause"


@dataclass(frozen=True, slots=True)
class SetValue:
    """Override `name` with `value`: one value for every row, or a list of one per row."""

    name: str
    value: Any
    kind: Literal["set"] = "set"


@dataclass(frozen=True, slots=True)
class Rewind:
    """Re-run from the node at `path`, keeping the current values upstream of it."""

    path: str
    kind: Literal["rewind"] = "rewind"


Command = Union[BreakAt, ClearBreak, StepOver, StepInto, Resume, Pause, SetValue, Rewind]
