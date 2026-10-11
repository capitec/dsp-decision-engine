"""Sessions: signing out, protected routes, an expired session during editing and the background refresh."""

import os
import re
import time

import pytest
from playwright.sync_api import Page, expect

from flat_rules_ui import open_flat_rules
from platform_ui import edit_is_stored, open_projects, open_with_edit, press_save, toast
from seed import create_project
from test_flat_rule_params import flat_document
from tree_ui import sign_in, sign_out

EXPIRED_DIALOG = "Session Expired"


def refresh_session_through_the_popup(page: Page, email: str) -> None:
    """Session Expired dialog -> Refresh Session -> sign in again in the popup window."""
    with page.expect_popup() as popup:
        page.get_by_role("button", name="Refresh Session").click()
    sign_in(popup.value, email)
    expect(toast(page, "Session Restored")).to_be_visible()


# ---- signing out ---------------------------------------------------------------------------------------------------

@pytest.mark.smoke
def test_the_back_button_after_signing_out_does_not_show_a_protected_page(signed_in: Page):
    open_projects(signed_in)
    sign_out(signed_in)

    signed_in.go_back()

    expect(signed_in.get_by_role("button", name="Sign In")).to_be_visible()
    expect(signed_in.get_by_role("heading", name="All Projects")).to_have_count(0)
    expect(signed_in.get_by_role("button", name="Overview")).to_have_count(0)


def test_the_back_button_after_signing_out_of_an_editor_does_not_show_the_project(signed_in: Page):
    project = create_project(signed_in, "Private editor", flat_document())["id"]
    open_flat_rules(signed_in, project)
    signed_in.goto("/projects")
    sign_out(signed_in)

    signed_in.evaluate("history.go(-2)")  # past the project list, to the editor's own history entry

    expect(signed_in.get_by_role("button", name="Sign In")).to_be_visible()
    expect(signed_in.get_by_role("button", name="Add Rule")).to_have_count(0)
    expect(signed_in.get_by_text("Private editor")).to_have_count(0)


def test_api_calls_after_signing_out_are_refused(signed_in: Page):
    project = create_project(signed_in, "Mine", flat_document())["id"]
    sign_out(signed_in)

    assert signed_in.request.get("/api/v1/projects").status == 401
    assert signed_in.request.get(f"/api/v1/projects/{project}").status == 401
    assert signed_in.request.put(f"/api/v1/projects/{project}", data={"name": "x"}).status == 401


@pytest.mark.parametrize("route", ["/", "/teams", "/profile", "/api-tokens", "/datasets",
                                   "/projects/00000000-0000-4000-8000-000000000000",
                                   "/flat-rules/00000000-0000-4000-8000-000000000000"])
def test_a_signed_out_visitor_is_sent_to_login_from_every_protected_route(page: Page, route: str):
    page.goto(route)

    expect(page).to_have_url(re.compile(r"/login$"))
    expect(page.get_by_role("button", name="Sign In")).to_be_visible()


def test_signing_in_again_after_signing_out_shows_only_the_users_own_projects(signed_in: Page, email: str):
    create_project(signed_in, "Mine again", flat_document())
    sign_out(signed_in)

    sign_in(signed_in, email)

    expect(signed_in.get_by_role("button", name="Overview")).to_be_visible()
    open_projects(signed_in)
    expect(signed_in.get_by_text("Mine again", exact=True)).to_be_visible()


# ---- an expired session ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["flat", "tree"])
def test_saving_with_an_expired_session_says_so_and_keeps_the_edit_on_screen(signed_in: Page, kind: str):
    open_with_edit(signed_in, kind)
    signed_in.context.clear_cookies()

    press_save(signed_in)

    expect(signed_in.get_by_role("dialog").get_by_text(EXPIRED_DIALOG)).to_be_visible()
    if kind == "flat":
        expect(signed_in.get_by_test_id("rule-card")).to_have_count(2)
    else:
        expect(signed_in.get_by_text("Unsaved changes")).to_be_visible()
    signed_in.get_by_role("button", name="Sign Out & Lose Progress").click()
    expect(signed_in.get_by_role("button", name="Sign In")).to_be_visible()


def test_an_idle_editor_notices_the_expired_session_by_itself(signed_in: Page):
    open_with_edit(signed_in, "flat")
    signed_in.context.clear_cookies()

    expect(signed_in.get_by_role("dialog").get_by_text(EXPIRED_DIALOG)).to_be_visible(timeout=20_000)


@pytest.mark.parametrize("kind", ["flat", "tree"])
def test_the_save_interrupted_by_the_expired_session_goes_through_after_the_refresh(signed_in: Page, email: str, kind: str):
    if kind == "flat" and os.environ.get("E2E_LIVE_COLLAB") == "true":
        # The edit is already stored over the live socket, which cookies cleared in the browser do not end, so no save
        # is interrupted; test_live_collab's sign-out test covers a live session that ends.
        pytest.skip("live editing stores the edit before Save")
    project = open_with_edit(signed_in, kind)
    signed_in.context.clear_cookies()
    press_save(signed_in)
    expect(signed_in.get_by_role("dialog").get_by_text(EXPIRED_DIALOG)).to_be_visible()

    refresh_session_through_the_popup(signed_in, email)

    expect(toast(signed_in, "Project Saved")).to_be_visible()
    assert edit_is_stored(signed_in, project, kind)


def closed_popup_without_signing_in(page: Page) -> None:
    open_with_edit(page, "flat")
    page.context.clear_cookies()
    press_save(page)
    expect(page.get_by_role("dialog").get_by_text(EXPIRED_DIALOG)).to_be_visible()
    with page.expect_popup() as popup:
        page.get_by_role("button", name="Refresh Session").click()
    popup.value.close()


def test_closing_the_refresh_popup_without_signing_in_keeps_the_dialog_and_the_edit(signed_in: Page):
    closed_popup_without_signing_in(signed_in)

    expect(signed_in.get_by_role("button", name="Refresh Session")).to_be_enabled()
    expect(signed_in.get_by_role("dialog").get_by_text(EXPIRED_DIALOG)).to_be_visible()
    expect(signed_in.get_by_test_id("rule-card")).to_have_count(2)


@pytest.mark.xfail(strict=True, reason="closing the refresh popup without signing in shows no reason: the check for 'cancelled' reads a stale isRefreshing")
def test_closing_the_refresh_popup_without_signing_in_says_what_happened(signed_in: Page):
    closed_popup_without_signing_in(signed_in)

    expect(signed_in.get_by_role("dialog").get_by_text("Authentication was cancelled or failed")).to_be_visible()


# ---- the background refresh -------------------------------------------------------------------------------------------

def set_session_expiry_in(page: Page, seconds: int) -> None:
    """Move the session_status cookie (the client's view of the expiry) to `seconds` from now."""
    cookie = next(c for c in page.context.cookies() if c["name"] == "session_status")
    page.context.add_cookies([{**cookie, "value": str(int(time.time()) + seconds)}])


def test_editing_near_the_end_of_the_session_asks_the_server_to_renew_it(signed_in: Page):
    open_with_edit(signed_in, "flat", "Near the end")
    set_session_expiry_in(signed_in, 300)

    with signed_in.expect_request(re.compile(r"/api/auth/config")):
        signed_in.get_by_role("button", name="Add Rule").click()


def test_editing_far_from_the_end_of_the_session_does_not_ask_for_a_renewal(signed_in: Page):
    open_with_edit(signed_in, "flat", "Fresh session")
    asked = []
    signed_in.on("request", lambda request: asked.append(request.url) if "/api/auth/config" in request.url else None)

    signed_in.get_by_role("button", name="Add Rule").click()
    expect(signed_in.get_by_test_id("rule-card")).to_have_count(3)

    assert asked == []
