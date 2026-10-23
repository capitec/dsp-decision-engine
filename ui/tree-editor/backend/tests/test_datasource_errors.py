import uuid

import pytest
from fastapi.testclient import TestClient

from dsp_decikit.api.v1.datasources.datasources import get_datasource_manager
from dsp_decikit.asgi import app
from dsp_decikit.auth.authorization import get_current_user_session
from dsp_decikit.auth.models import SessionInfo
from dsp_decikit.data_sources.core import DataSourceError

SECRET = "postgresql://admin:hunter2@db.internal:5432"


class FailingSource:
    def __init__(self, error):
        self.error = error

    async def preview_sql_query(self, **_):
        raise self.error

    async def upload_file(self, **_):
        raise self.error


class FailingManager:
    def __init__(self, error):
        self.error = error

    def get_datasource(self, _name):
        return FailingSource(self.error)

    async def get_datasource_tables(self, *_, **__):
        raise self.error

    async def preview_table_data(self, *_, **__):
        raise self.error


@pytest.fixture
def client_with():
    def make(error):
        app.dependency_overrides[get_current_user_session] = lambda: SessionInfo(user_id=uuid.uuid4())
        app.dependency_overrides[get_datasource_manager] = lambda: FailingManager(error)
        return TestClient(app)
    yield make
    app.dependency_overrides.clear()


def test_an_unexpected_error_listing_tables_is_not_sent_to_the_browser(client_with):
    response = client_with(RuntimeError(SECRET)).get("/api/v1/datasources/pg/tables")
    assert response.status_code == 500
    assert "hunter2" not in response.text and "Failed to list tables" in response.text


def test_a_data_source_error_keeps_its_message_for_the_user(client_with):
    response = client_with(DataSourceError("SQL query execution failed: no column amount")).post(
        "/api/v1/datasources/pg/preview-query", data={"source_table": "t", "sql_query": "select amount from t"})
    assert response.status_code == 500
    assert "no column amount" in response.json()["detail"]


def test_an_unexpected_error_in_a_streamed_preview_is_not_sent_to_the_browser(client_with):
    response = client_with(RuntimeError(SECRET)).get("/api/v1/datasources/pg/tables/t/preview-stream")
    assert '"type": "error"' in response.text and "hunter2" not in response.text


def test_an_unexpected_error_creating_a_view_is_not_sent_to_the_browser(client_with):
    response = client_with(RuntimeError(SECRET)).post(
        "/api/v1/datasources/create-view",
        data={"source_datasource": "pg", "dest_datasource": "pg", "source_table": "t", "view_name": "v",
              "sql_query": "select 1"})
    assert response.json()["success"] is False and "hunter2" not in response.text
