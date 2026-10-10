"""Editor behaviour around unsaved work, export, keyboard use, layout and the project rename dialog."""

import re

import pytest
from playwright.sync_api import Page, expect

from flat_docs import APPROVE, open_doc
from flat_rules_ui import export, fill_number, rename_rule, rule_card, rule_names, save, unary_rule
from seed import get_project

SMALL = {"type": "unary", "condition": {"type": "unary", "op": "<=", "feature": "amount", "threshold": 5}, "then": APPROVE}
UNSAVED = re.compile(r"unsaved|not saved", re.I)


def dialog_types(page: Page) -> list[str]:
    seen: list[str] = []
    page.on("dialog", lambda dialog: seen.append(dialog.type))
    return seen


@pytest.mark.xfail(strict=True, reason="the flat-rule editor has no unsaved-changes indicator")
def test_an_unsaved_edit_is_indicated(signed_in: Page):
    open_doc(signed_in, SMALL)
    expect(signed_in.get_by_text(UNSAVED)).to_have_count(0)
    rename_rule(signed_in, "Edited")

    expect(signed_in.get_by_text(UNSAVED).first).to_be_visible()


@pytest.mark.xfail(strict=True, reason="Back leaves the editor with unsaved edits without asking")
def test_back_after_an_unsaved_edit_asks_before_leaving(signed_in: Page):
    project = open_doc(signed_in, SMALL)
    seen = dialog_types(signed_in)
    rename_rule(signed_in, "Edited")
    signed_in.get_by_role("button", name="Back").click()

    assert seen or signed_in.url.endswith(project) or signed_in.get_by_text(UNSAVED).count()


@pytest.mark.xfail(strict=True, reason="closing the page with unsaved edits shows no beforeunload prompt")
def test_closing_the_tab_after_an_unsaved_edit_asks_before_leaving(signed_in: Page):
    open_doc(signed_in, SMALL)
    seen = dialog_types(signed_in)
    rename_rule(signed_in, "Edited")
    signed_in.close(run_before_unload=True)

    assert "beforeunload" in seen


def test_reloading_after_an_unsaved_edit_shows_the_saved_state(signed_in: Page):
    open_doc(signed_in, SMALL)
    rename_rule(signed_in, "Edited")
    signed_in.reload()

    expect(signed_in.get_by_test_id("rule-card")).to_have_count(1, timeout=30_000)
    assert rule_names(signed_in) == ["Rule 1"]


def test_saved_edit_is_still_there_after_leaving_and_opening_the_project_again(signed_in: Page):
    project = open_doc(signed_in, SMALL)
    rename_rule(signed_in, "Edited")
    save(signed_in)
    signed_in.get_by_role("button", name="Back").click()
    expect(signed_in.get_by_role("heading", name="Overview")).to_be_visible()  # Back lands on the overview
    signed_in.goto(f"/flat-rules/{project}")

    expect(signed_in.get_by_test_id("rule-card")).to_have_count(1, timeout=30_000)
    assert rule_names(signed_in) == ["Edited"]
    assert get_project(signed_in, project)["project_data"]["rules"][0]["meta"]["name"] == "Edited"


@pytest.mark.xfail(strict=True, reason="after Back, the browser's back button bounces off /flat-rules to the overview")
def test_browser_back_after_the_back_button_returns_to_the_editor(signed_in: Page):
    project = open_doc(signed_in, SMALL)
    signed_in.get_by_role("button", name="Back").click()
    expect(signed_in.get_by_role("heading", name="Overview")).to_be_visible()
    signed_in.go_back()

    expect(signed_in).to_have_url(re.compile(rf"/flat-rules/{project}$"))


@pytest.mark.xfail(strict=True, reason="Export downloads the stored document without the unsaved edit and says nothing")
def test_export_after_an_unsaved_edit_includes_it_or_says_it_was_left_out(signed_in: Page, tmp_path):
    open_doc(signed_in, SMALL)
    rename_rule(signed_in, "Edited")
    exported = export(signed_in, tmp_path)

    told = signed_in.get_by_role("status").filter(has_text=UNSAVED).count() > 0
    assert exported["rules"][0]["meta"]["name"] == "Edited" or told


def test_export_after_saving_includes_the_edit(signed_in: Page, tmp_path):
    open_doc(signed_in, SMALL)
    rename_rule(signed_in, "Edited")
    save(signed_in)

    assert export(signed_in, tmp_path)["rules"][0]["meta"]["name"] == "Edited"


def test_escape_closes_the_parameters_dialog_without_saving_a_new_row(signed_in: Page):
    open_doc(signed_in, SMALL)
    signed_in.get_by_role("button", name=re.compile(r"^Manage Parameters")).click()
    dialog = signed_in.get_by_role("dialog", name="Manage Parameters")
    dialog.get_by_role("button", name="Add Your First Parameter").click()
    dialog.get_by_placeholder("parameter_name").fill("draft")
    signed_in.keyboard.press("Escape")  # first closes the new row, the second the dialog
    expect(dialog.get_by_placeholder("parameter_name")).to_have_count(0)
    signed_in.keyboard.press("Escape")

    expect(dialog).to_be_hidden()
    assert save(signed_in)["project_data"]["parameters"] == {}


def test_enter_in_the_new_parameter_name_adds_the_row(signed_in: Page):
    open_doc(signed_in, SMALL)
    signed_in.get_by_role("button", name=re.compile(r"^Manage Parameters")).click()
    dialog = signed_in.get_by_role("dialog", name="Manage Parameters")
    dialog.get_by_role("button", name="Add Your First Parameter").click()
    dialog.get_by_placeholder("parameter_name").fill("limit")
    dialog.get_by_placeholder("parameter_name").press("Enter")

    expect(dialog.get_by_placeholder("parameter_name")).to_have_count(0)  # the new-row form is gone
    expect(dialog.locator("tbody tr").get_by_role("textbox").first).to_have_value("limit")
    dialog.get_by_role("button", name="Save Changes").click()
    assert save(signed_in)["project_data"]["parameters"]["limit"]["type"] == "String"


def test_enter_in_the_parameter_picker_chooses_the_first_suggestion(signed_in: Page):
    open_doc(signed_in, SMALL, parameters={"limit": {"type": "Float64", "default_value": 1}})
    card = rule_card(signed_in)
    card.get_by_role("button", name="#").first.click()
    box = card.get_by_placeholder("parameter_name")
    box.fill("lim")
    box.press("Enter")

    expect(box).to_have_value("limit")
    assert save(signed_in)["project_data"]["rules"][0]["rule"]["condition"]["threshold"] == {"param": "limit"}


def test_escape_in_the_parameter_picker_closes_the_suggestions_and_keeps_the_text(signed_in: Page):
    open_doc(signed_in, SMALL, parameters={"limit": {"type": "Float64", "default_value": 1}})
    card = rule_card(signed_in)
    card.get_by_role("button", name="#").first.click()
    box = card.get_by_placeholder("parameter_name")
    box.fill("lim")
    expect(card.get_by_text("Manage Parameters")).to_be_visible()
    box.press("Escape")

    expect(card.get_by_text("Manage Parameters")).to_be_hidden()
    expect(box).to_have_value("lim")


@pytest.mark.xfail(strict=True, reason="Enter and Escape do nothing in the rule rename box; only the check button ends it")
def test_enter_in_the_rule_rename_box_finishes_the_rename(signed_in: Page):
    open_doc(signed_in, SMALL)
    card = rule_card(signed_in)
    card.get_by_test_id("rule-rename").click()
    card.get_by_role("textbox").first.fill("Edited")
    card.get_by_role("textbox").first.press("Enter")

    expect(card.get_by_test_id("rule-name")).to_have_text("Edited")


def test_tab_moves_from_the_threshold_to_its_parameter_button(signed_in: Page):
    open_doc(signed_in)
    card = unary_rule(signed_in, "<=")
    card.get_by_test_id("numeric-input").first.locator("input").focus()
    signed_in.keyboard.press("Tab")

    expect(card.get_by_role("button", name="#").first).to_be_focused()


NO_WIDENING = "document.documentElement.scrollWidth <= document.documentElement.clientWidth"


def test_a_huge_threshold_does_not_widen_the_page(signed_in: Page):
    open_doc(signed_in, SMALL)
    fill_number(rule_card(signed_in), "123456789012345678901234567890")

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert float(reply["project_data"]["rules"][0]["rule"]["condition"]["threshold"]) == 1.2345678901234568e29
    assert signed_in.evaluate(NO_WIDENING)


@pytest.mark.xfail(strict=True, reason="an unbroken 150-character rule name widens the page")
def test_a_very_long_rule_name_does_not_widen_the_page(signed_in: Page):
    open_doc(signed_in, SMALL)
    rename_rule(signed_in, "a-very-long-unbroken-name-" + "x" * 150)

    assert signed_in.evaluate(NO_WIDENING)


def test_a_very_long_rule_description_does_not_widen_the_page(signed_in: Page):
    open_doc(signed_in, SMALL)
    card = rule_card(signed_in)
    card.get_by_text("Add description...").click()
    card.get_by_placeholder("Add description...").fill("d" * 300)

    assert signed_in.evaluate(NO_WIDENING)


def test_project_rename_dialog_updates_the_name_and_description(signed_in: Page):
    project = open_doc(signed_in, SMALL, name="Before")
    signed_in.get_by_role("heading", level=1).locator("xpath=following-sibling::button").click()
    dialog = signed_in.get_by_role("dialog", name="Edit Project Details")
    expect(dialog.get_by_label("Project Name")).to_have_value("Before")
    dialog.get_by_label("Project Name").fill("After")
    dialog.get_by_label("Description").fill("What it decides")
    dialog.get_by_role("button", name="Save Changes").click()

    expect(signed_in.get_by_role("heading", level=1)).to_have_text("After")
    expect(signed_in.get_by_text("What it decides")).to_be_visible()
    saved = get_project(signed_in, project)
    assert (saved["name"], saved["description"]) == ("After", "What it decides")


def test_project_rename_dialog_refuses_an_empty_name_and_cancel_changes_nothing(signed_in: Page):
    project = open_doc(signed_in, SMALL, name="Keep")
    pencil = signed_in.get_by_role("heading", level=1).locator("xpath=following-sibling::button")
    pencil.click()
    dialog = signed_in.get_by_role("dialog", name="Edit Project Details")
    dialog.get_by_label("Project Name").fill("   ")
    expect(dialog.get_by_role("button", name="Save Changes")).to_be_disabled()
    dialog.get_by_label("Project Name").fill("Dropped")
    dialog.get_by_role("button", name="Cancel").click()

    expect(signed_in.get_by_role("heading", level=1)).to_have_text("Keep")
    assert get_project(signed_in, project)["name"] == "Keep"
