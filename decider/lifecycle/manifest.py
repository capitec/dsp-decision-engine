from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict

from decider.contract.refs import FlowRef, RecordRef
from decider.exceptions import DeciderError

MANIFEST_VERSION = "1"
"""The version of the run-manifest format, bumped like `CONTRACT_VERSION`: removing
or renaming a field is a major change; adding an optional one is not. Manifests carry
this in `RunManifest.manifest_version` and readers reject a newer one (see
`check_manifest_version`)."""


class Revision(BaseModel):
    """A source revision as authored and as resolved for a run.

    `authored` is the symbolic intent (`"main"`, `"v2.1.0"`); `resolved` is the
    immutable git SHA the run actually executed, `None` when the tree was dirty
    or the revision could not be resolved.

    Example::

        Revision(authored="main", resolved="6a9a9e")
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    authored: str
    resolved: str | None = None


class SourceState(BaseModel):
    """The state of the source the run read from.

    `clean` says the tree had no uncommitted changes; a saved reproducible run
    requires a clean committed revision. `fingerprint` is a content hash over
    the scope `fingerprint_scope` names, so a live graph can detect that its
    source moved on while a manifest keeps the fingerprint it was captured with.

    Example::

        SourceState(clean=True, fingerprint="ab12", fingerprint_scope="closure")
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    clean: bool = True
    dirty_paths: tuple[str, ...] = ()
    fingerprint: str | None = None
    fingerprint_scope: str = ""


class Environment(BaseModel):
    """The engine, Python and dependency environment of a run.

    `python` is the interpreter version, `decider` the engine version, and
    `dependencies` the resolved third-party versions (e.g. `"polars==1.0.0"`).

    Example::

        Environment(python="3.11.4", decider="0.1.0", dependencies=("polars==1.0.0",))
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    python: str = ""
    decider: str = ""
    dependencies: tuple[str, ...] = ()


class InputFingerprint(BaseModel):
    """Identity and shape of the input data: dataset, schema, row count, content fingerprint.

    `columns` lists `(column, dtype)` pairs in column order, which is part of
    identity; `fingerprint` is the content hash that binds the exact data.

    Example::

        InputFingerprint(dataset="loans.parquet", fingerprint="ab12", columns=(("client_id", "str"),), row_count=1000)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str = ""
    fingerprint: str | None = None
    columns: tuple[tuple[str, str], ...] = ()
    row_count: int | None = None


class Sample(BaseModel):
    """A declared sampling selection, reproducible through its `seed`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fraction: float | None = None
    count: int | None = None
    seed: int | None = None


class Selection(BaseModel):
    """Which records a run selected: a filter, a sampling, and explicit record references.

    `records` reuses `RecordRef`, so record identity is the contract's durable
    key rather than a session row ordinal.

    Example::

        Selection(filter="sector_code == 1", sample=Sample(fraction=0.1, seed=7),
                  records=(RecordRef(dataset="loans.parquet", key={"client_id": "C-1"}),))
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    filter: str | None = None
    sample: Sample | None = None
    records: tuple[RecordRef, ...] = ()


class Override(BaseModel):
    """A declared override applied to a run: a value written over `target`.

    `target` is a value slot spec like `ValueSlotRef.spec` (`"name@path"` or a
    bare input name), `value` the declared replacement.

    Example::

        Override(target="term_cap@term/cap_by_income", value=36.0)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str
    value: Any


class ResultRef(BaseModel):
    """A reference to a result the client owns, not the bulk data itself.

    `location` is where the client stored the result (a path or URI),
    `fingerprint` the content hash that binds it, `row_count` its size when known.

    Example::

        ResultRef(name="trace", location="file:///runs/m1/trace.jsonl", fingerprint="ab12")
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    location: str = ""
    fingerprint: str | None = None
    row_count: int | None = None


class RunManifest(BaseModel):
    """An immutable, versioned record of one run, enough to reopen or share it.

    The manifest is small metadata: bulk results and traces stay in the client's
    storage and are reached through `outputs`. It is immutable so evidence stays
    bound to the source fingerprint it was captured with; a later source edit
    makes the live graph stale but never rewrites this manifest. `reproducible`
    is true only for a clean committed revision.

    Example::

        manifest = RunManifest(
            manifest_id="m1",
            kind="experiment_run",
            flow=FlowRef(name="term", source="credit.pipeline:build"),
            revision=Revision(authored="main", resolved="6a9a9e"),
            input=InputFingerprint(dataset="loans.parquet", fingerprint="ab12", row_count=1000),
        )
        manifest.reproducible  # True
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_version: str = MANIFEST_VERSION
    manifest_id: str
    kind: str
    flow: FlowRef | None = None
    revision: Revision | None = None
    source: SourceState = SourceState()
    environment: Environment = Environment()
    input: InputFingerprint | None = None
    selection: Selection = Selection()
    overrides: tuple[Override, ...] = ()
    outputs: tuple[ResultRef, ...] = ()

    @property
    def reproducible(self) -> bool:
        return self.revision is not None and self.revision.resolved is not None and self.source.clean


def new_manifest_id() -> str:
    """A fresh opaque manifest id, so a result never depends on an ephemeral session id."""
    return uuid.uuid4().hex


def check_manifest_version(version: str | None) -> None:
    """Raise if `version` is newer than this decider supports; mirrors `check_version`.

    A manifest without a version, or with an older one, is accepted and migrated
    forward on read; a newer one means a reader that doesn't know the format, so
    it must refuse rather than guess.

    Example::

        check_manifest_version(manifest.manifest_version)
    """
    if version is None:
        return
    try:
        given = int(version.partition(".")[0])
        supported = int(MANIFEST_VERSION.partition(".")[0])
    except ValueError:
        raise DeciderError(f"unrecognised manifest version {version!r}") from None
    if given > supported:
        raise DeciderError(
            f"manifest version {version} is newer than {MANIFEST_VERSION}; upgrade decider to read it"
        )


def source_is_stale(captured: str | None, current: str) -> bool:
    """Whether source has moved on since `captured` was fingerprinted.

    `None` means no fingerprint was captured (unknown, not stale); a mismatch
    means the live graph, breakpoints or resolved steps should re-resolve, while
    any manifest bound to `captured` stays as it is.

    Example::

        source_is_stale(manifest.source.fingerprint, fingerprint_now)
    """
    return captured is not None and captured != current


def json_schema() -> dict[str, Any]:
    """The versioned JSON Schema of the run-manifest contract, generated from the Python models.

    `RunManifest` and every model it names appear under `$defs` (the reused
    contract `FlowRef` and `RecordRef` included); `manifest_version` marks the
    schema version.

    Example::

        schema = json_schema()
        schema["manifest_version"]  # "1"
    """
    schema = RunManifest.model_json_schema()
    schema["manifest_version"] = MANIFEST_VERSION
    return schema
