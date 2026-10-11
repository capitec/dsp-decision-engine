import uuid

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from dsp_decikit.api.v1.project.project import ProjectUpdate
from dsp_decikit.asgi import app
from dsp_decikit.auth.authorization import get_current_user_session
from dsp_decikit.auth.models import SessionInfo


@pytest.fixture
def client():
    app.dependency_overrides[get_current_user_session] = lambda: SessionInfo(user_id=uuid.uuid4())
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_a_document_save_without_base_version_is_refused_with_422_naming_the_field(client):
    # Refused before the handler runs, so no database is needed.
    reply = client.put(f"/api/v1/projects/{uuid.uuid4()}", json={"project_data": {"type": "tree"}})

    assert reply.status_code == 422
    assert "base_version" in reply.json()["validation_errors"][0]["msg"]


@pytest.mark.parametrize("body", [
    {"project_data": {"type": "tree"}, "base_version": 3},
    {"name": "Renamed"},
    {"description": "d", "folder_id": str(uuid.uuid4())},
    {"name": "Renamed", "project_data": None},
])
def test_a_versioned_document_save_or_a_metadata_only_update_is_accepted(body):
    ProjectUpdate.model_validate(body)


def test_a_document_save_with_a_null_base_version_is_refused():
    with pytest.raises(ValidationError, match="base_version"):
        ProjectUpdate.model_validate({"project_data": {}, "base_version": None})
