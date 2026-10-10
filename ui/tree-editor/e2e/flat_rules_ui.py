"""Flat-rule editor actions: each does what a user does, nothing more."""

import json
import re

from playwright.sync_api import Locator, Page, expect


def open_flat_rules(page: Page, project_id: str) -> None:
    """Open the flat-rule editor on a project and wait until it has loaded."""
    page.goto(f"/flat-rules/{project_id}")
    expect(page.get_by_role("button", name="Add Rule")).to_be_visible(timeout=30_000)  # the test machine is shared


def rule_card(page: Page, rule: int = 0) -> Locator:
    """The card of the `rule`-th rule, in the order shown."""
    return page.get_by_test_id("rule-card").nth(rule)


def open_parameters(page: Page) -> Locator:
    """Manage Parameters; returns the dialog."""
    page.get_by_role("button", name=re.compile(r"^Manage Parameters")).click()
    return page.get_by_role("dialog", name="Manage Parameters")


def add_parameter(page: Page, name: str, dtype: str = "Float64", value: str | None = None) -> None:
    """Manage Parameters -> new row -> name, type, default -> Add -> Save Changes."""
    dialog = open_parameters(page)
    first = dialog.get_by_role("button", name="Add Your First Parameter")
    (first if first.is_visible() else dialog.get_by_role("button", name="Add Parameter")).click()
    row = dialog.locator("tbody tr").filter(has=page.get_by_placeholder("parameter_name"))
    row.get_by_placeholder("parameter_name").fill(name)
    if dtype != "String":
        row.get_by_role("combobox").click()
        page.get_by_role("option", name=dtype, exact=True).click()
    if value is not None:
        row.locator("td").nth(2).locator("input").first.fill(value)
    row.get_by_role("button", name="Add", exact=True).click()
    dialog.get_by_role("button", name="Save Changes").click()
    expect(dialog).to_be_hidden()


def set_parameter_default(page: Page, name: str, value: str) -> None:
    """Manage Parameters -> the row of `name` -> new default -> Save Changes."""
    dialog = open_parameters(page)
    row = dialog.locator("tbody tr").filter(has=page.locator(f"input[value='{name}']"))
    row.locator("td").nth(2).locator("input").first.fill(value)
    dialog.get_by_role("button", name="Save Changes").click()
    expect(dialog).to_be_hidden()


def set_node_type(page: Page, scope: Locator, kind: str, index: int = 0) -> None:
    """Leaf, Unary, Cases or Composite for the `index`-th node of `scope`."""
    scope.get_by_test_id("node-type").nth(index).click()
    page.get_by_role("option", name=kind, exact=True).click()


def choose_feature(page: Page, scope: Locator, feature: str) -> None:
    """Pick `feature` in the first feature selector of `scope`."""
    scope.get_by_text("Select or search feature...").first.click()
    page.get_by_role("option", name=feature, exact=True).click()


def choose_output(page: Page, scope: Locator, label: str, index: int = 0) -> None:
    """Pick the output record whose label contains `label` in the `index`-th leaf of `scope`."""
    scope.get_by_test_id("leaf-output").nth(index).get_by_role("combobox").click()
    page.get_by_role("option").filter(has_text=label).first.click()


def use_parameter(scope: Locator, name: str, index: int = 0) -> None:
    """Click the `index`-th # (next to a value), then type the parameter's name."""
    toggle = scope.get_by_role("button", name="#").nth(index)
    holder = toggle.locator("xpath=..")
    toggle.click()
    box = holder.get_by_placeholder("parameter_name")
    box.fill(name)
    box.press("Escape")


def fill_parameter(scope: Locator, name: str, index: int = 0) -> None:
    """Retype the parameter name of the `index`-th parameter field that is already showing."""
    box = scope.get_by_placeholder("parameter_name").nth(index)
    box.fill(name)
    box.press("Escape")


def save(page: Page) -> dict:
    """Click Save; returns the server's reply (with `validation_errors`)."""
    with page.expect_response(lambda r: r.request.method == "PUT" and "/api/v1/projects/" in r.url) as reply:
        page.get_by_role("button", name="Save", exact=True).click()
    assert reply.value.ok, reply.value.text()
    return reply.value.json()


def export(page: Page, tmp_path) -> dict:
    """Click Export and return the downloaded document; fails if the export is refused."""
    with page.expect_download() as download:
        page.get_by_role("button", name="Export").click()
    path = tmp_path / "export.json"
    download.value.save_as(path)
    return json.loads(path.read_text())


OPERATOR_LABELS = {"<=": "≤", ">=": "≥", "between": "Between", "isin": "IsIn", "string_match": "String Match",
                   "is_null": "Is Null", "is_not_null": "Is Not Null", "is_true": "Is True", "is_false": "Is False",
                   "and": "AND", "or": "OR", "not": "NOT"}
MATCH_LABELS = {"exact": "Exact", "starts_with": "Starts With", "contains": "Contains", "ends_with": "Ends With",
                "regex": "Regex"}


def _pick(page: Page, trigger: Locator, label: str) -> None:
    trigger.click()
    page.get_by_role("option", name=label, exact=True).click()


def choose_operator(page: Page, scope: Locator, op: str, index: int = 0, testid: str = "unary-op") -> None:
    """Pick `op` ("<=", "between", "isin", ...) in the `index`-th operator selector of `scope`."""
    _pick(page, scope.get_by_test_id(testid).nth(index), OPERATOR_LABELS.get(op, op))


def choose_condition_op(page: Page, scope: Locator, op: str, index: int = 0) -> None:
    """Pick an operator or AND/OR/NOT in the `index`-th row of a composite's condition list."""
    choose_operator(page, scope, op, index, testid="condition-op")


def choose_match_type(page: Page, scope: Locator, match_type: str, index: int = 0) -> None:
    """Pick a string match type ("exact", "regex", ...) in the `index`-th match-type selector of `scope`."""
    _pick(page, scope.get_by_test_id("match-type").nth(index), MATCH_LABELS[match_type])


def number_input(scope: Locator, index: int = 0) -> Locator:
    """The `index`-th numeric field (threshold, bound) of `scope`."""
    return scope.get_by_test_id("numeric-input").nth(index).locator("input")


def fill_number(scope: Locator, text: str, index: int = 0) -> None:
    """Type `text` into the `index`-th numeric field of `scope`."""
    number_input(scope, index).fill(text)


def add_isin_values(scope: Locator, values: list[int | str]) -> None:
    """Type each value into the last empty IsIn chip field of `scope`, Enter after each."""
    # the placeholder disappears with the first chip, so type into the focused field instead of re-locating it
    scope.get_by_placeholder("Type integers and press Enter").last.click()
    for value in values:
        scope.page.keyboard.type(str(value))
        scope.page.keyboard.press("Enter")


def fill_pattern(scope: Locator, text: str, index: int = 0) -> None:
    """Type `text` into the `index`-th string-match pattern of `scope`."""
    scope.get_by_placeholder(re.compile(r"^Pattern \d+$")).nth(index).fill(text)


def add_then(page: Page, scope: Locator, output: str | None = "approve", index: int = 0) -> None:
    """Add the `index`-th THEN branch of `scope` and point its leaf at `output` (None keeps the default row)."""
    scope.get_by_role("button", name="Add THEN").nth(index).click()
    if output:
        choose_output(page, scope, output, index)


def add_otherwise(scope: Locator, index: int = 0) -> None:
    """Add the `index`-th OTHERWISE branch of `scope` (a leaf on the default row)."""
    scope.get_by_role("button", name="Add OTHERWISE").nth(index).click()


def unary_rule(page: Page, op: str, *, feature: str = "amount", then: str | None = "approve",
               otherwise: bool = False) -> Locator:
    """Rule 1 becomes `<feature> <op> ...` -> THEN `then` (and an OTHERWISE default row when asked); returns the card."""
    card = rule_card(page)
    set_node_type(page, card, "Unary")
    choose_feature(page, card, feature)
    if op != "<=":
        choose_operator(page, card, op)
    add_then(page, card, then)
    if otherwise:
        add_otherwise(card)
    return card


def rename_rule(page: Page, new_name: str, rule: int = 0) -> None:
    """Rename the `rule`-th rule with its pencil and check buttons."""
    card = rule_card(page, rule)
    card.get_by_test_id("rule-rename").click()
    box = card.get_by_role("textbox").first
    box.fill(new_name)
    card.get_by_test_id("rule-rename-done").click()


def rule_names(page: Page) -> list[str]:
    """The names of the rule cards shown, in order."""
    return page.get_by_test_id("rule-name").all_inner_texts()


def open_tab(page: Page, name: str) -> None:
    """Rules, Configure Inputs or Configure Outputs."""
    page.get_by_role("tab", name=re.compile(name)).click()


def build_nested_composite(page: Page, card: Locator, ops: list[str], feature: str = "amount",
                           threshold: str = "100") -> None:
    """Rule `card` becomes ops[0](ops[1](... <feature> <= threshold)), one condition per level."""
    set_node_type(page, card, "Composite")
    choose_operator(page, card, ops[0], testid="composite-op")
    card.get_by_role("button", name="Add Condition", exact=True).click()
    for op in ops[1:]:
        row = card.get_by_test_id("condition-row").last
        choose_condition_op(page, row, op)
        row.get_by_role("button", name="Add Nested Condition").click()
    row = card.get_by_test_id("condition-row").last
    choose_feature(page, row, feature)
    fill_number(row, threshold)


def add_output_column(page: Page, name: str, dtype: str | None = None) -> None:
    """Configure Outputs -> Add Field -> name (and type) -> Add."""
    page.get_by_role("button", name="Add Field").click()
    box = page.get_by_placeholder("Column name")
    box.fill(name)
    if dtype:
        box.locator("xpath=..").get_by_role("combobox").click()
        page.get_by_role("option", name=re.compile(rf"^{re.escape(dtype)}\b")).first.click()
    page.get_by_role("button", name="Add", exact=True).click()


def output_header(page: Page, column: str) -> Locator:
    """The header cell of output column `column` on the Configure Outputs tab."""
    return page.locator("thead th").filter(has_text=column)


def rename_output_column(page: Page, old: str, new: str) -> None:
    """Click the column name, type the new one, Enter."""
    output_header(page, old).get_by_title("Click to rename").click()
    box = page.locator("thead").get_by_role("textbox")
    box.fill(new)
    box.press("Enter")


def delete_output_column(page: Page, column: str) -> None:
    """Delete field (the native confirm is accepted by the signed_in fixture)."""
    output_header(page, column).get_by_title("Delete field").click()


def set_output_column_type(page: Page, column: str, dtype: str) -> None:
    """Pick `dtype` in the type selector of output column `column`."""
    output_header(page, column).get_by_role("combobox").click()
    page.get_by_role("option", name=dtype, exact=True).click()


def output_cell(page: Page, record: int, column: int) -> Locator:
    """The editable field of output record `record` (table row), column index `column`."""
    return page.locator("tbody tr").nth(record).locator("td").nth(column).locator("input, [role=combobox]").first


def open_custom_types(page: Page) -> Locator:
    """Manage Custom Types; returns the dialog."""
    page.get_by_role("button", name=re.compile(r"^Manage Custom Types")).click()
    return page.get_by_role("dialog", name="Manage Custom Types")


def add_categorical_type(page: Page, name: str, categories: list[str]) -> None:
    """In the open Manage Custom Types dialog: New Categorical -> name, categories -> Save Type."""
    dialog = page.get_by_role("dialog", name="Manage Custom Types")
    dialog.get_by_role("button", name="New Categorical").first.click()
    dialog.get_by_placeholder("e.g., ActionType, OutcomeCategories").fill(name)
    dialog.get_by_placeholder("Add category...").click()
    for category in categories:
        page.keyboard.type(category)
        page.keyboard.press("Enter")
    dialog.get_by_role("button", name="Save Type").click()
