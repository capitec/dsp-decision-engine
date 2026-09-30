from __future__ import annotations

import pytest

from decider.contract import FlowRef, RecordRef
from decider.exceptions import DeciderError
from decider.lifecycle import (MANIFEST_VERSION, InputFingerprint, Override, ResultRef, Revision, RunManifest,
                               check_manifest_version, json_schema, new_manifest_id, source_is_stale)


def make_manifest(**fields) -> RunManifest:
    defaults = dict(
        manifest_id="m1",
        kind="experiment_run",
        flow=FlowRef(name="term", source="credit.pipeline:build"),
        revision=Revision(authored="main", resolved="6a9a9e"),
        input=InputFingerprint(dataset="loans.parquet", fingerprint="ab12", row_count=1000),
    )
    defaults.update(fields)
    return RunManifest(**defaults)


def test_manifest_is_immutable():
    manifest = make_manifest()
    with pytest.raises(Exception):
        manifest.kind = "check_run"


def test_reproducible_requires_clean_resolved_revision():
    assert make_manifest().reproducible
    assert not make_manifest(revision=Revision(authored="main", resolved=None)).reproducible
    assert not make_manifest(source={"clean": False}).reproducible
    assert not make_manifest(revision=None).reproducible


def test_revision_keeps_authored_intent_and_resolved_sha():
    revision = Revision(authored="main", resolved="6a9a9e")
    assert revision.authored == "main"
    assert revision.resolved == "6a9a9e"


def test_selection_reuses_record_ref():
    manifest = make_manifest(selection={
        "filter": "sector_code == 1",
        "records": (RecordRef(dataset="loans.parquet", key={"client_id": "C-1"}),),
    })
    assert manifest.selection.filter == "sector_code == 1"
    assert manifest.selection.records[0].durable


def test_check_manifest_version_rejects_a_newer_format():
    check_manifest_version(None)
    check_manifest_version(MANIFEST_VERSION)
    with pytest.raises(DeciderError):
        check_manifest_version(str(int(MANIFEST_VERSION) + 1))


def test_manifest_round_trips_through_json_without_a_session_id():
    manifest = make_manifest(outputs=(ResultRef(name="trace", location="file:///runs/m1/trace.jsonl", fingerprint="ab12"),))
    restored = RunManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest
    assert restored.manifest_id == "m1"
    assert restored.revision.resolved == "6a9a9e"
    assert restored.outputs[0].name == "trace"


def test_source_is_stale_only_when_a_fingerprint_differs():
    assert source_is_stale("ab12", "cd34")
    assert not source_is_stale("ab12", "ab12")
    assert not source_is_stale(None, "cd34")


def test_json_schema_is_versioned_and_self_contained():
    schema = json_schema()
    assert schema["manifest_version"] == MANIFEST_VERSION
    assert schema["title"] == "RunManifest"
    assert "FlowRef" in schema.get("$defs", {})
    assert "Revision" in schema.get("$defs", {})


def test_new_manifest_id_is_unique():
    assert new_manifest_id() != new_manifest_id()


def test_overrides_declare_a_target_and_value():
    override = Override(target="term_cap@term/cap_by_income", value=36.0)
    assert override.target == "term_cap@term/cap_by_income"
    assert override.value == 36.0
