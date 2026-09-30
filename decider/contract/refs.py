from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from decider.exceptions import DeciderError

CONTRACT_VERSION = "1"
"""The version of the static-flow contract. Bump it when a change breaks readers of
older documents: removing or renaming a field is a major change; adding an optional
one is not. Documents carry this in `FlowDescription.contract_version` and readers
reject a version newer than they support (see `check_version`)."""


class FlowRef(BaseModel):
    """A whole pipeline, the root of a flow description.

    `flow_id` is the durable identity a committed source declares (see the ID
    command); it is `None` until then. `name`, `source` and `revision` are
    capture-time description kept for repair when the durable id no longer
    resolves. `source` is the import path of the step that produced the root,
    `revision` the resolved source revision (a git SHA) when one is known.

    Example::

        FlowRef(name="term", source="credit.pipeline:build")
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    flow_id: str | None = None
    name: str = ""
    source: str | None = None
    revision: str | None = None


class StepRef(BaseModel):
    """One node of a flow.

    The durable global reference is `(flow_id, step_id)`: both come from
    committed source, so a trace or finding links to a step that survives
    renames and reordering. `path` is the derived path (`Origin.path`), unique
    within one build but only development context. `source` is the import path
    of the code that produced the node, `locator` a position inside a `row`
    node. `inputs`/`outputs` are the value names the node reads/writes
    (`kind` is the call kind for a call node).

    Example::

        StepRef(path="term/cap_by_income", source="credit.rules:cap_by_income", node_type="call", kind="scalar")
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    flow_id: str | None = None
    step_id: str | None = None
    path: str
    name: str = ""
    source: str | None = None
    locator: str | None = None
    node_type: str | None = None
    kind: Literal["scalar", "row", "frame"] | None = None
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()


class EdgeRef(BaseModel):
    """An edge between two nodes of one flow.

    `kind="control"` is structural: `to_path` runs inside `from_path`. `kind="data"`
    is a dependency: `to_path` reads each name in `values` from the node at
    `from_path`. Edges name nodes by path; the nodes themselves are the
    `FlowDescription.nodes` entries.

    Example::

        EdgeRef(from_path="affordability", to_path="affordability/ratio", kind="control")
        EdgeRef(from_path="affordability/ratio", to_path="affordability/affordable", kind="data", values=("ratio",))
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_path: str
    to_path: str
    kind: Literal["control", "data"] = "control"
    values: tuple[str, ...] = ()


class ValueSlotRef(BaseModel):
    """A static value slot: one written version of a value, `name` produced at `path`.

    This is the place a value is read from or written to, not a runtime value:
    the value observed during a run and the trace evidence of that run are
    separate concepts owned by the run/debug layer. `path` is `None` for a
    pipeline input column. `spec` spells the slot as `State.versions` reads it:
    `"name"` or `"name@path"`.

    Example::

        ValueSlotRef(name="term_cap", path="term/cap_by_income").spec  # "term_cap@term/cap_by_income"
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    path: str | None = None
    flow_id: str | None = None

    @property
    def spec(self) -> str:
        return self.name if self.path is None else f"{self.name}@{self.path}"


class RecordRef(BaseModel):
    """A reference to one record of a data source.

    A durable reference needs an explicit record key: `key` maps the key
    columns to the values that identify the record, and is empty (not durable)
    until a caller supplies a validated one. `ordinal` is a row number within
    one session and is display-only: it is never part of identity and must not
    be persisted, because row order is not stable across loads.

    Example::

        RecordRef(dataset="loans.parquet", key={"client_id": "C-1182"})
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str
    key: dict[str, Any] = {}
    ordinal: int | None = None

    @property
    def durable(self) -> bool:
        return bool(self.key)


class FlowDescription(BaseModel):
    """The static structure of one flow: its nodes, edges and value slots.

    Static only. Runtime facts — the values a run observed, the params it
    validated, the events a session emitted — live on `RunReport` and
    `Session.events`, not here. `contract_version` is the contract the document
    was written against.

    Example::

        from decider.contract import describe
        desc = describe(pipeline)
        desc.nodes[0].path
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_version: str = CONTRACT_VERSION
    flow: FlowRef
    nodes: tuple[StepRef, ...]
    edges: tuple[EdgeRef, ...]
    value_slots: tuple[ValueSlotRef, ...]


def check_version(version: str | None) -> None:
    """Raise if `version` is newer than this decider supports.

    A document without a version, or with an older one, is accepted and
    migrated forward on read; a newer one means a reader that doesn't know the
    format, so it must refuse rather than guess.

    Example::

        check_version(desc.contract_version)
    """
    if version is None:
        return
    try:
        given = int(version.partition(".")[0])
        supported = int(CONTRACT_VERSION.partition(".")[0])
    except ValueError:
        raise DeciderError(f"unrecognised contract version {version!r}") from None
    if given > supported:
        raise DeciderError(
            f"contract version {version} is newer than {CONTRACT_VERSION}; upgrade decider to read it"
        )
