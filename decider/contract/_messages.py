"""The editor bridge's reply payloads, as the models `tools/decider-ui/src/model/protocol.ts` is generated from.

A field a reply leaves out is left unset when built, and `wire()` dumps only what was set, so the
JSON on the wire is exactly what the producer put in. Regenerate the TypeScript with
`uv run python -m decider.contract._messages`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from decider.contract._describe import Capabilities
from decider.contract._manifest import InputFingerprint
from decider.contract._schema import interfaces

PROTOCOL_VERSION = "1"

START, END = "// <generated: uv run python -m decider.contract._messages>", "// </generated>"
PROTOCOL_TS = Path(__file__).parents[2] / "tools/decider-ui/src/model/protocol.ts"

Number = int | float


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
    doc: str = Field(None, description="The first line of the step's docstring.")
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
    modifies: list[str] = None
    carries: list[str] = None
    maxIterations: int = None
    children: list[CallNodeJson | GroupNodeJson]
    folded: list[str] = Field(None, description="Set on a group drawn folded: the paths of the steps inside it.")


class PipelineCandidate(_Msg):
    name: str
    line: int | None
    kind: str
    status: Literal["pipeline", "invalid"] = Field(None, description="`invalid` candidates are shown with their reason instead of being offered.")
    reason: str = None


class ParamInfo(_Open):
    type: str
    default: Any = None
    required: bool = None
    ge: Number = None
    le: Number = None
    gt: Number = None
    lt: Number = None


class FieldMeta(_Open):
    kind: str
    precision: int = None
    scale: int = None
    units: str = None
    unit: str = None
    assumed: bool = Field(None, description="Not declared: guessed from the value's name.")


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
    values: dict[str, Any] = Field(None, description="The module's `PARAMS` document: the values the flow runs with (tables' rows included).")
    outcome: list[str] = Field(None, description="What the flow decides: its emitted values and what its top-level branches set.")
    valuesFile: str | None = Field(None, description="The file `PARAMS` is read from, when the module names one.")
    fields: dict[str, FieldMeta] = Field(None, description="Each value's declared metadata, by name; the rest are guessed (`fieldOf`).")


class Checkpoint(_Msg):
    path: str
    when: Literal["before", "after"]
    iteration: int = Field(None, description="Inside a loop: which iteration.")


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
    origin: SessionOrigin = None


class Hit(_Msg):
    watch: int
    text: str
    path: str = None
    rows: list[int] = None
    values: list[Any] = None


class TraceStatus(_Msg):
    available: bool
    reason: str = None


class Status(_Msg):
    finished: bool
    current: Checkpoint | None
    events: list[SessionEvent]
    error: str = None
    hit: Hit | None = Field(None, description="Set when a value or iteration breakpoint paused the run.")
    ran: list[str] = Field(None, description="Every step and group that has run since the start or the last rewind.")
    trace: TraceStatus = Field(None, description="Whether structured decision trace is captured for this run.")
    dataset: InputFingerprint | None = Field(None, description="Identity of the data the run started on, as `InputFingerprint` dumps it.")


class ColumnSummary(Summary):
    name: str
    value: Any = Field(description="The focused record's value, when a record is focused.")
    producer: str
    versions: int
    writtenBy: str = Field(None, description='For the focused record: the last step that wrote the value, and the last that changed it ("input" for neither).')
    changedBy: str = None


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
    via: Literal["merge", "carry"] = None
    inputs: list[Lineage]
    forcedRows: list[int] = Field(None, description="Records a comparison forced; the others ran as they are.")
    setBy: str = Field(None, description='Set when the value came from you: "force@<branch or loop>" or "override@<path>", and what it was before.')
    was: Any = None
    unset: bool = Field(None, description="Not set yet for the focused record: there is nothing to break down.")
    missing: bool = Field(None, description="Nothing has written this value yet.")
    arms: list[str] = Field(None, description="For a forced branch condition: its arms by name, so a value reads as the arm it picks.")


class DraftOverride(_Msg):
    target: str
    value: Any
    source: str = Field(description='`"set"` (a value overridden mid-run) or `"force"` (a branch/loop forced).')
    row: int | None = None


class DraftDrop(_Msg):
    kind: str
    path: str = None
    detail: str = None


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
