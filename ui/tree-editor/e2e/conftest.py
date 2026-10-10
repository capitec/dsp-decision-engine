import os
import signal
import uuid

import pytest
from playwright.sync_api import Page

from tree_ui import register


def pytest_configure(config):
    # A run stopped by its PID (SIGTERM) still unwinds its teardowns, so no `decikit serve` it started outlives it.
    signal.signal(signal.SIGTERM, signal.default_int_handler)


def pytest_collection_modifyitems(items):
    for item in items:
        if not any(item.get_closest_marker(m) for m in ("smoke", "serve", "slow")):
            item.add_marker(pytest.mark.full)


@pytest.fixture(scope="session")
def base_url():
    return os.environ.get("E2E_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    # the editor sidebar needs a desktop-sized viewport
    return {**browser_context_args, "viewport": {"width": 1500, "height": 950}, "accept_downloads": True}


@pytest.fixture
def email() -> str:
    return f"{uuid.uuid4().hex[:10]}@example.com"


@pytest.fixture
def signed_in(page: Page, email: str) -> Page:
    page.on("dialog", lambda dialog: dialog.accept())  # "leave with unsaved changes?"
    register(page, email)
    return page
