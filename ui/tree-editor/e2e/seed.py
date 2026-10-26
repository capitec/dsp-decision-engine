"""State set up through the HTTP API the UI uses, with the signed-in page's cookies: fast setup, not behaviour under test."""

import json

from playwright.sync_api import Page


def create_project(page: Page, name: str, document: dict | None = None, project_type: str = "flat_rule",
                   folder_id: str | None = None) -> dict:
    """POST /api/v1/projects; returns the created project (with `id` and `validation_errors`)."""
    body = {"name": name, "project_type": project_type, "project_data": document, "folder_id": folder_id}
    r = page.request.post("/api/v1/projects", data={k: v for k, v in body.items() if v is not None})
    assert r.ok, r.text()
    return r.json()


def import_document(page: Page, name: str, document: dict | str, filename: str = "tree.json") -> dict:
    """POST /api/v1/projects/import, as the Import Project dialog does; returns the reply."""
    content = document if isinstance(document, str) else json.dumps(document)
    r = page.request.post("/api/v1/projects/import", multipart={
        "file": {"name": filename, "mimeType": "application/json", "buffer": content.encode()},
        "name": name,
    })
    assert r.ok, r.text()
    return r.json()


def get_project(page: Page, project_id: str) -> dict:
    r = page.request.get(f"/api/v1/projects/{project_id}")
    assert r.ok, r.text()
    return r.json()


def export_document(page: Page, project_id: str) -> dict:
    """GET /api/v1/projects/<id>/export; fails if the server refuses the export."""
    r = page.request.get(f"/api/v1/projects/{project_id}/export")
    assert r.ok, r.text()
    return r.json()


def create_folder(page: Page, name: str, parent_id: str | None = None, description: str | None = None) -> dict:
    """POST /api/v1/folders; returns the created folder."""
    body = {"name": name, "parent_id": parent_id, "description": description}
    r = page.request.post("/api/v1/folders", data={k: v for k, v in body.items() if v is not None})
    assert r.ok, r.text()
    return r.json()


def list_folders(page: Page) -> list[dict]:
    r = page.request.get("/api/v1/folders")
    assert r.ok, r.text()
    return r.json()


def list_projects(page: Page, **query) -> dict:
    """GET /api/v1/projects (a page of results: `items`, `total`, `pages`)."""
    r = page.request.get("/api/v1/projects", params=query)
    assert r.ok, r.text()
    return r.json()


def my_id(page: Page) -> str:
    r = page.request.get("/api/v1/user/profile")
    assert r.ok, r.text()
    return r.json()["id"]


def share_project(page: Page, project_id: str, user_id: str, access_mode: str = "read") -> dict:
    """Share a project with one user."""
    r = page.request.post(f"/api/v1/projects/{project_id}/shares",
                          data={"share_type": "user", "user_id": user_id, "access_mode": access_mode})
    assert r.ok, r.text()
    return r.json()


def share_folder(page: Page, folder_id: str, user_id: str, access_mode: str = "read") -> dict:
    """Share a folder (and everything in it) with one user."""
    r = page.request.post(f"/api/v1/folders/{folder_id}/shares",
                          data={"share_type": "user", "user_id": user_id, "access_mode": access_mode})
    assert r.ok, r.text()
    return r.json()
