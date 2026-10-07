import re
import uuid

from playwright.sync_api import Browser, Page, expect

from tree_ui import create_tree_project, register, sign_in, sign_out


def test_sign_out_then_sign_in_again(signed_in: Page, email: str):
    sign_out(signed_in)
    sign_in(signed_in, email)
    expect(signed_in.get_by_role("button", name="Overview")).to_be_visible()


def test_wrong_password_is_rejected(signed_in: Page, email: str):
    sign_out(signed_in)
    sign_in(signed_in, email, password="not-the-password")
    expect(signed_in.get_by_role("button", name="Sign In")).to_be_visible()
    expect(signed_in.get_by_role("button", name="Overview")).to_be_hidden()


def test_signed_out_visitor_is_sent_to_login(page: Page):
    page.goto("/projects")
    expect(page).to_have_url(re.compile(r"/login$"))


def test_new_user_overview_counts_only_their_projects(signed_in: Page, browser: Browser, base_url: str):
    other = browser.new_context(base_url=base_url).new_page()
    register(other, f"{uuid.uuid4().hex[:10]}@example.com")
    create_tree_project(other, "Someone else's tree")
    other.context.close()

    signed_in.reload()
    expect(signed_in.get_by_text("Total Projects").locator("xpath=following-sibling::*")).to_have_text("0")
