"""Page actions shared by the tests: each does what a user does, nothing more."""

from playwright.sync_api import Page, expect

PASSWORD = "Passw0rd!x"


def register(page: Page, email: str) -> None:
    page.goto("/")
    page.get_by_role("button", name="Need an account? Register").click()
    page.locator("input[type=text]").fill("E2E User")
    page.locator("input[type=email]").fill(email)
    page.locator("input[type=password]").first.fill(PASSWORD)
    page.get_by_role("button", name="Create Account").click()
    expect(page.get_by_role("button", name="Overview")).to_be_visible()


def sign_out(page: Page) -> None:
    page.get_by_role("button", name="EU").click()
    page.get_by_role("menuitem", name="Sign out").click()
    expect(page.get_by_role("button", name="Sign In")).to_be_visible()


def sign_in(page: Page, email: str, password: str = PASSWORD) -> None:
    page.locator("input[type=email]").fill(email)
    page.locator("input[type=password]").fill(password)
    page.get_by_role("button", name="Sign In").click()


def create_tree_project(page: Page, name: str) -> None:
    """Wizard: one Float64 feature `amount` and one String output `decision`."""
    page.get_by_role("button", name="Projects").click()
    page.get_by_role("button", name="Create New Project").click()
    page.get_by_role("menuitem", name="Create Tree Project").click()
    page.get_by_placeholder("Enter project name").fill(name)
    page.get_by_role("button", name="Next").click()
    page.get_by_role("button", name="Add Field").first.click()
    page.get_by_placeholder("Field name").fill("amount")
    page.get_by_role("combobox").filter(has_text="String").click()
    page.get_by_role("option", name="Float64", exact=True).click()
    page.get_by_role("button", name="Next").click()
    page.get_by_role("button", name="Add Field").first.click()
    page.get_by_placeholder("Field name").fill("decision")
    page.get_by_role("button", name="Create Project").click()
    page.wait_for_url("**/projects/*-*")
    expect(page.get_by_role("button", name="Add Node")).to_be_visible()


def _drag(page: Page, node, dx: int, dy: int) -> None:
    box = node.bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + dx, y + dy, steps=10)
    page.mouse.up()


def _connect(page: Page, parent, child) -> None:
    a = parent.locator(".react-flow__handle-bottom").bounding_box()
    t = child.locator(".react-flow__handle-top").bounding_box()
    page.mouse.move(a["x"] + a["width"] / 2, a["y"] + a["height"] / 2)
    page.mouse.down()
    page.mouse.move(t["x"] + t["width"] / 2, t["y"] + t["height"] / 2, steps=10)
    page.mouse.up()


def build_loan_tree(page: Page, limit: int = 1000) -> None:
    """`amount <= limit` -> approve, otherwise decline."""
    page.get_by_role("button", name="Edit Output").click()
    for _ in range(2):
        page.get_by_role("button", name="Add Record").click()
    for field, text in zip(page.locator("[role=dialog] input").all(), ["review", "approve", "decline"]):
        field.fill(text)
    page.get_by_role("button", name="Close").click()

    for _ in range(3):
        page.get_by_role("button", name="Add Node").click()
    nodes = page.locator(".react-flow__node")
    expect(nodes).to_have_count(3)
    # new nodes land on top of each other; spread them out so the handles are reachable
    _drag(page, nodes.nth(2), 150, 200)
    _drag(page, nodes.nth(1), -300, 200)
    _drag(page, nodes.nth(0), 0, -250)

    nodes.nth(0).click()
    page.get_by_role("combobox").filter(has_text="Leaf Node").click()
    page.get_by_role("option", name="Unary Condition").click()
    page.get_by_text("Select or search feature...").click()
    page.get_by_role("option", name="amount").click()
    page.get_by_placeholder("0").fill(str(limit))

    _connect(page, nodes.nth(0), nodes.nth(1))  # first edge: condition holds
    _connect(page, nodes.nth(0), nodes.nth(2))  # second edge: otherwise
    expect(page.locator(".react-flow__edge")).to_have_count(2)

    for index, outcome in ((1, "approve"), (2, "decline")):
        nodes.nth(index).click()
        page.get_by_role("combobox").filter(has_text="-1").click()
        page.get_by_role("option", name=outcome).click()


def save(page: Page) -> None:
    page.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Unsaved changes")).to_be_hidden()
