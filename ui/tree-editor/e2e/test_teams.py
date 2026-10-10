"""Teams: the teams page, who may create one, and sharing a project or a folder with a team."""

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from platform_db import create_team
from platform_fixtures import Other, other  # noqa: F401  (the fixture)
from platform_ui import open_projects, project_action, project_card, toast
from seed import create_project, my_id
from test_flat_rule_params import flat_document


def team_name() -> str:
    return f"team-{uuid.uuid4().hex[:8]}"


def share_with_team(page: Page, project: str, team: str, access: str = "Read Only") -> None:
    """Share dialog of a project card -> Teams tab -> search by name -> Share."""
    project_action(page, project, "Share")
    dialog = page.get_by_role("dialog")
    dialog.get_by_role("tab", name="Teams").click()
    dialog.get_by_placeholder("Search by team name...").fill(team)
    if access != "Read Only":
        dialog.get_by_role("combobox").click()
        page.get_by_role("option", name=access, exact=True).click()
    dialog.get_by_role("button", name="Share", exact=True).click()
    expect(dialog.get_by_role("tab", name="Manage Shares (1)")).to_be_visible()


def test_the_teams_page_of_a_new_user_says_they_belong_to_none(signed_in: Page):
    signed_in.goto("/teams")

    expect(signed_in.get_by_text("No teams yet")).to_be_visible()


def test_a_regular_user_creating_a_team_is_told_it_failed_and_no_team_appears(signed_in: Page):
    signed_in.goto("/teams")
    signed_in.get_by_role("button", name="Create Team").click()
    signed_in.get_by_placeholder("Enter team name").fill("Mine")

    signed_in.get_by_role("button", name="Create Team").last.click()

    expect(toast(signed_in, "Error")).to_be_visible()
    expect(signed_in.get_by_text("No teams yet")).to_be_visible()


@pytest.mark.xfail(strict=True, reason="the toast for a refused team creation says 'Failed to create team'; the server's reason (only superadmins can create teams) is dropped, and the page offers Create Team to everyone")
def test_a_regular_user_is_told_why_a_team_cannot_be_created(signed_in: Page):
    signed_in.goto("/teams")
    signed_in.get_by_role("button", name="Create Team").click()
    signed_in.get_by_placeholder("Enter team name").fill("Mine")

    signed_in.get_by_role("button", name="Create Team").last.click()

    expect(signed_in.get_by_text(re.compile("superadmin", re.I)).first).to_be_visible()


def test_a_team_the_user_belongs_to_is_listed_with_their_role(signed_in: Page, other: Other):
    name = team_name()
    create_team(name, my_id(signed_in), (other.id,))

    other.page.goto("/teams")

    card = other.page.get_by_role("heading", name=name).locator("xpath=ancestor::div[contains(@class,'cursor-pointer')][1]")
    expect(card).to_contain_text("member")
    signed_in.goto("/teams")
    expect(signed_in.get_by_role("heading", name=name)).to_be_visible()


def test_a_project_shared_with_a_team_appears_for_its_members_only(signed_in: Page, other: Other):
    name = team_name()
    create_team(name, my_id(signed_in), (other.id,))
    create_project(signed_in, "For the team", flat_document())
    open_projects(signed_in)

    share_with_team(signed_in, "For the team", name)

    open_projects(other.page)
    expect(project_card(other.page, "For the team")).to_be_visible()


def test_a_project_shared_with_a_team_is_not_listed_for_non_members(signed_in: Page, other: Other):
    name = team_name()
    create_team(name, my_id(signed_in))
    project = create_project(signed_in, "Members only", flat_document())["id"]
    team = signed_in.request.get("/api/v1/teams/search", params={"q": name}).json()["teams"][0]["id"]
    assert signed_in.request.post(f"/api/v1/projects/{project}/shares", data={"share_type": "team", "team_id": team, "access_mode": "read"}).ok

    open_projects(other.page)

    expect(project_card(other.page, "Members only")).to_have_count(0)


def test_removing_the_team_share_takes_the_project_away_from_its_members(signed_in: Page, other: Other):
    name = team_name()
    create_team(name, my_id(signed_in), (other.id,))
    create_project(signed_in, "Revoked from team", flat_document())
    open_projects(signed_in)
    share_with_team(signed_in, "Revoked from team", name)
    open_projects(other.page)
    expect(project_card(other.page, "Revoked from team")).to_be_visible()

    dialog = signed_in.get_by_role("dialog")
    dialog.get_by_role("tab", name=re.compile("Manage Shares")).click()
    dialog.get_by_test_id("share-remove").click()
    expect(dialog.get_by_test_id("share-row")).to_have_count(0)

    open_projects(other.page)
    expect(project_card(other.page, "Revoked from team")).to_have_count(0)
