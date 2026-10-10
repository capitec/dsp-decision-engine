"""A second signed-in user in a browser context of their own, for the sharing and permission tests."""

import uuid
from typing import NamedTuple

import pytest
from playwright.sync_api import Browser, Page

from seed import my_id
from tree_ui import register


class Other(NamedTuple):
    page: Page
    email: str
    id: str


@pytest.fixture
def other(browser: Browser, base_url: str):
    context = browser.new_context(base_url=base_url, accept_downloads=True, viewport={"width": 1500, "height": 950})
    page = context.new_page()
    page.on("dialog", lambda dialog: dialog.accept())
    email = f"{uuid.uuid4().hex[:10]}@example.com"
    register(page, email)
    yield Other(page, email, my_id(page))
    context.close()
