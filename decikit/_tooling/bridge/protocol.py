"""The editor bridge's reply payloads, as the models `ui/lib/src/model/protocol.ts` is generated from.

A field a reply leaves out is left unset when built, and `wire()` dumps only what was set, so the
JSON on the wire is exactly what the producer put in. Regenerate the TypeScript with
`uv run python -m decikit._tooling.bridge.protocol`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from decikit._tooling.bridge.schema import interfaces
from decikit.lab import CONTRACT_VERSION
from decikit.lab.data import InputFingerprint

PROTOCOL_VERSION = "1"

START, END = "// <generated: uv run python -m decikit._tooling.bridge.protocol>", "// </generated>"
PROTOCOL_TS = Path(__file__).parents[3] / "ui/lib/src/model/protocol.ts"

Number = int | float

_T = TypeVar("_T")
# Absent from the wire when unset, never `null`: None for Python, left out of the generated TypeScript.
Unset = Union[_T, SkipJsonSchema[None]]


# Optional extra (pyproject) -> the module that proves it is installed.
_OPTIONAL_EXTRAS = {
    "serve-starlette": "uvicorn",
    "serve-sanic": "sanic",
    "notebook": "IPython",
    "mcp": "fastmcp",
    "prove": "z3",
}


class Capabilities(BaseModel):
    """What this decikit build can do, for callers that must degrade gracefully.

    `modes` lists the registered engines a debug session can run on (the
    `NumbaEngine` tags). `optional_dependencies` maps each optional extra to
    whether it is installed (unavailable ones raise
    `DecikitMissingDependencyError` on use). `source_mapping` says whether an
    import-path source can be resolved to a file and line.

    Example::

        capabilities().modes  # ("fused", "interpreted", "stepped")
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_version: str = CONTRACT_VERSION
    modes: tuple[str, ...]
    optional_dependencies: dict[str, bool]
    source_mapping: bool = True


def capabilities() -> Capabilities:
    """The capabilities of the installed decikit."""
    from decikit import Engine, NumbaEngine  # imports numba

    return Capabilities(
        modes=tuple(sorted(tag for tag, cls in Engine.registered().items() if issubclass(cls, NumbaEngine))),
        optional_dependencies={extra: importlib.util.find_spec(module) is not None
                               for extra, module in _OPTIONAL_EXTRAS.items()},
    )


class _Msg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def wire(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class _Open(_Msg):
    model_config = ConfigDict(extra="allow")


class Hello(_Msg):
    protocol: str
    contract_version: str
    capabilities: Capabilities


class PythonLocation(_Msg):
    file: str
    line: int
    bodyLine: int | None
    endLine: int | None = None


class CallNodeJson(_Msg):
    path: str
    source: str
    file: str | None
    line: int | None
    id: str | None = Field(None, description="The step's durable id (12 hex), surviving renames; absent or `null` for a step that never got one.")
    step_id: str | None = Field(None, description="The contract's `StepRef.step_id`: the same durable id as `id`.")
    kind: Literal["call"]
    callKind: Literal["scalar", "row", "frame"]
    inputs: list[str] | None = Field(description="`null`: a frame step of unknown lineage.")
    outputs: list[str] | None
    params: dict[str, Any]
    python: PythonLocation | None = Field(description="The Python that runs in interpreted mode: `fn`, or a row node's `reference`. `bodyLine` is its first statement past a docstring, `endLine` its last line.")
    code: str = Field(description="Changes when the code or config behind the node changes.")
    doc: Unset[str] = Field(None, description="The first line of the step's docstring.")
    formula: str | None = Field(None, description="What a one-line step returns, e.g. `min(pl_raw_rate, repo_rate + cap_margin)`.")
    body: str | None = Field(None, description="The step's source, when short.")
    formulaBefore: str | None = Field(None, description="Set once the step's code was edited mid-run: the formula it had before.")
    table: dict[str, Any] | None = Field(None, description='A lookup table\'s match, e.g. `{type: "between", variable, lower_bound_column, upper_bound_column}`.')


class GroupNodeJson(_Msg):
    path: str
    source: str
    file: str | None
    line: int | None
    id: str | None = Field(None, description="The step's durable id (12 hex), surviving renames; absent or `null` for a step that never got one.")
    step_id: str | None = Field(None, description="The contract's `StepRef.step_id`: the same durable id as `id`.")
    kind: Literal["sequence", "branch", "loop"]
    modifies: Unset[list[str]] = None
    carries: Unset[list[str]] = None
    maxIterations: Unset[int] = None
    children: list[CallNodeJson | GroupNodeJson]
    folded: Unset[list[str]] = Field(None, description="Set on a group drawn folded: the paths of the steps inside it.")


class PipelineCandidate(_Msg):
    name: str
    line: int | None
    kind: str
    status: Unset[Literal["pipeline", "invalid"]] = Field(None, description="`invalid` candidates are shown with their reason instead of being offered.")
    reason: Unset[str] = None


class ParamInfo(_Open):
    type: str
    default: Any = None
    required: Unset[bool] = None
    ge: Unset[Number] = None
    le: Unset[Number] = None
    gt: Unset[Number] = None
    lt: Unset[Number] = None


class FieldMeta(_Open):
    kind: str
    precision: Unset[int] = None
    scale: Unset[int] = None
    units: Unset[str] = None
    unit: Unset[str] = None
    assumed: Unset[bool] = Field(None, description="Not declared: guessed from the value's name.")


class FlowSize(_Msg):
    nodes: int
    calls: int


class DescribeResult(_Msg):
    pipelines: list[PipelineCandidate]
    pipeline: str
    flow_id: str | None = Field(None, description="The contract's `FlowRef.flow_id`: the durable id of the assembled flow.")
    size: FlowSize = Field(description="The assembled entry-point flow: its node and step counts.")
    ir: CallNodeJson | GroupNodeJson
    params: dict[str, dict[str, ParamInfo]] = Field(description="Node path (or \"shared\") -> param name -> its type, default and bounds.")
    values: Unset[dict[str, Any]] = Field(None, description="The module's `PARAMS` document: the values the flow runs with (tables' rows included).")
    outcome: Unset[list[str]] = Field(None, description="What the flow decides: its emitted values and what its top-level branches set.")
    valuesFile: str | None = Field(None, description="The file `PARAMS` is read from, when the module names one.")
    fields: Unset[dict[str, FieldMeta]] = Field(None, description="Each value's declared metadata, by name; the rest are guessed (`fieldOf`).")


class Checkpoint(_Msg):
    path: str
    when: Literal["before", "after"]
    iteration: Unset[int] = Field(None, description="Inside a loop: which iteration.")


class Summary(_Msg):
    dtype: str
    rows: int
    nulls: int
    preview: list[Any]


class SessionOrigin(_Open):
    path: str
    source: str
    locator: str | None


class SessionEvent(_Open):
    kind: str
    origin: Unset[SessionOrigin] = None


class Hit(_Msg):
    watch: int
    text: str
    path: Unset[str] = None
    rows: Unset[list[int]] = None
    values: Unset[list[Any]] = None


class TraceStatus(_Msg):
    available: bool
    reason: Unset[str] = None


class Status(_Msg):
    finished: bool
    current: Checkpoint | None
    events: list[SessionEvent]
    error: Unset[str] = None
    hit: Hit | None = Field(None, description="Set when a value or iteration breakpoint paused the run.")
    ran: Unset[list[str]] = Field(None, description="Every step and group that has run since the start or the last rewind.")
    trace: Unset[TraceStatus] = Field(None, description="Whether structured decision trace is captured for this run.")
    dataset: InputFingerprint | None = Field(None, description="Identity of the data the run started on, as `InputFingerprint` dumps it.")


class ColumnSummary(Summary):
    name: str
    value: Any = Field(description="The focused record's value, when a record is focused.")
    producer: str
    versions: int
    writtenBy: Unset[str] = Field(None, description='For the focused record: the last step that wrote the value, and the last that changed it ("input" for neither).')
    changedBy: Unset[str] = None


class KeyColumn(_Msg):
    name: str
    values: list[Any]


class StateResult(_Msg):
    columns: list[ColumnSummary]
    row: int | None
    key: KeyColumn | None


class Lineage(_Msg):
    name: str
    producer: str | None
    value: Any
    via: Unset[Literal["merge", "carry"]] = None
    inputs: list[Lineage]
    forcedRows: Unset[list[int]] = Field(None, description="Records a comparison forced; the others ran as they are.")
    setBy: Unset[str] = Field(None, description='Set when the value came from you: "force@<branch or loop>" or "override@<path>", and what it was before.')
    was: Any = None
    unset: Unset[bool] = Field(None, description="Not set yet for the focused record: there is nothing to break down.")
    missing: Unset[bool] = Field(None, description="Nothing has written this value yet.")
    arms: Unset[list[str]] = Field(None, description="For a forced branch condition: its arms by name, so a value reads as the arm it picks.")


class DraftOverride(_Msg):
    target: str
    value: Any
    source: str = Field(description='`"set"` (a value overridden mid-run) or `"force"` (a branch/loop forced).')
    row: int | None = None


class DraftDrop(_Msg):
    kind: str
    path: Unset[str] = None
    detail: Unset[str] = None


class Draft(_Msg):
    converted: list[DraftOverride]
    dropped: list[DraftDrop]


MODELS = (Hello, Capabilities, PythonLocation, CallNodeJson, GroupNodeJson, PipelineCandidate, DescribeResult, FlowSize, ParamInfo,
          Checkpoint, Summary, SessionOrigin, SessionEvent, Status, TraceStatus, Hit, InputFingerprint, ColumnSummary,
          StateResult, KeyColumn, Lineage, FieldMeta, Draft, DraftOverride, DraftDrop)

for _m in (GroupNodeJson, DescribeResult, Lineage):
    _m.model_rebuild()


def generated_block() -> str:
    return f"{START}\n{interfaces(MODELS)}{END}"


def render(current: str) -> str:
    """`protocol.ts`'s text with its generated block replaced by the one the models give."""
    head, _, rest = current.partition(START)
    return head + generated_block() + rest.partition(END)[2]


if __name__ == "__main__":
    PROTOCOL_TS.write_text(render(PROTOCOL_TS.read_text()))
    sys.stdout.write(f"wrote {PROTOCOL_TS}\n")
