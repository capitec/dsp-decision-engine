"""Live editing of one flat-rule project from several browsers, on one or two backend nodes.

Runs only against a stack started with DSP_LIVE_COLLAB=true (`stack.sh` passes it on as E2E_LIVE_COLLAB); the
two-node tests also need E2E_NODES=2 (E2E_BASE_URL_2).
"""

import http.client
import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest
from playwright.sync_api import Browser, Page, expect

from flat_docs import SMALL, document
from flat_rules_ui import open_flat_rules, rename_rule
from platform_fixtures import other  # noqa: F401
from platform_ui import press_save, toast
from seed import create_project, get_project, share_project
from tree_ui import sign_in

pytestmark = pytest.mark.skipif(os.environ.get("E2E_LIVE_COLLAB") != "true",
                                reason="needs a stack started with DSP_LIVE_COLLAB=true")
SECOND_NODE = os.environ.get("E2E_BASE_URL_2")
two_nodes = pytest.mark.skipif(not SECOND_NODE, reason="needs E2E_NODES=2")
LIVE = re.compile(r"/api/v1/projects/[0-9a-f-]{36}/live$")


class Held:
    """Holds the `ops` messages a page sends on its live socket while `holding`; `release()` sends them on."""

    def __init__(self, page: Page):
        self.holding, self.buffer = False, []
        page.route_web_socket(LIVE, self._route)

    def _route(self, ws):
        server = ws.connect_to_server()

        def outgoing(message):
            if self.holding and json.loads(message).get("type") == "ops":
                self.buffer.append((server, message))
            else:
                server.send(message)

        ws.on_message(outgoing)

    def release(self):
        self.holding = False
        for server, message in self.buffer:
            server.send(message)
        self.buffer = []

    def wait_held(self, page: Page, count: int = 1):
        deadline = time.monotonic() + 10
        while len(self.buffer) < count:
            assert time.monotonic() < deadline, "the edit was never sent"
            page.wait_for_timeout(50)


@pytest.fixture
def session(browser: Browser, base_url: str, signed_in: Page, email: str):
    """`session(node=1, hold=False)`: another browser signed in as the same user (on node 1 or 2) -> (page, Held)."""
    contexts = []

    def open_session(node: int = 1):
        context = browser.new_context(base_url=SECOND_NODE if node == 2 else base_url,
                                      viewport={"width": 1500, "height": 950})
        contexts.append(context)
        page = context.new_page()
        page.on("dialog", lambda dialog: dialog.accept())
        page.goto("/")
        sign_in(page, email)
        expect(page.get_by_role("button", name="Overview")).to_be_visible()
        return page

    yield open_session
    for context in contexts:
        context.close()


class Line:
    """A page's live socket through a switch: `cut()` closes it and refuses new ones, `deaden()` keeps it open but
    lets nothing through (a dead TCP connection), `restore()` lets sockets through again."""

    def __init__(self, page: Page):
        self.state, self.open = "up", []
        page.route_web_socket(LIVE, self._route)

    def _route(self, ws):
        if self.state == "cut":
            # Closing inside the route handler hangs Playwright; the page's hello comes right after it opens.
            ws.on_message(lambda _: ws.close())
            return
        server = ws.connect_to_server()
        ws.on_message(lambda m: self.state == "up" and server.send(m))
        server.on_message(lambda m: self.state == "up" and ws.send(m))
        self.open.append((ws, server))

    def cut(self):
        self.state = "cut"
        for ws, server in self.open:
            ws.close()
            server.close()
        self.open = []

    def deaden(self):
        self.state = "dead"

    def restore(self):
        # A socket that went dead stays dead, as a real one would; the next one gets through.
        self.cut()
        self.state = "up"


OFFLINE = "Offline - edits will be merged when you reconnect"


def two_rules(page: Page) -> str:
    return create_project(page, "Live", document(SMALL, SMALL))["id"]


def open_live(page: Page, project: str, hold: bool = False) -> Held | None:
    held = Held(page) if hold else None
    open_flat_rules(page, project)
    expect(page.get_by_test_id("live-status")).to_have_text("Live")
    return held


def stored(page: Page, project: str, check, timeout: float = 10) -> dict:
    """The stored document once `check(document)` holds (live writes land a moment after the edit)."""
    deadline = time.monotonic() + timeout
    while True:
        data = get_project(page, project)["project_data"]
        if check(data) or time.monotonic() > deadline:
            assert check(data), data
            return data
        page.wait_for_timeout(100)


def names(data: dict) -> list[str]:
    return [r["meta"]["name"] for r in data["rules"]]


def shows(page: Page, expected: list[str]):
    expect(page.get_by_test_id("rule-name")).to_have_text(expected)


def add_rule(page: Page):
    page.get_by_role("button", name="Add Rule").click()


# ---- one node, several browsers -------------------------------------------------------------------------------------

def test_an_edit_in_one_browser_shows_in_a_second_and_a_third(signed_in: Page, session):
    project = two_rules(signed_in)
    second, third = session(), session()
    for page in (signed_in, second, third):
        open_live(page, project)

    rename_rule(signed_in, "Renamed live")

    for page in (second, third):
        shows(page, ["Renamed live", "Rule 2"])
    stored(signed_in, project, lambda d: names(d) == ["Renamed live", "Rule 2"])


def test_simultaneous_edits_to_different_rules_are_all_kept(signed_in: Page, session):
    project = two_rules(signed_in)
    second = session()
    held = [open_live(page, project, hold=True) for page in (signed_in, second)]
    for h in held:
        h.holding = True

    rename_rule(signed_in, "From first", rule=0)
    rename_rule(second, "From second", rule=1)
    held[0].wait_held(signed_in)
    held[1].wait_held(second)
    for h in held:
        h.release()

    for page in (signed_in, second):
        shows(page, ["From first", "From second"])
    stored(signed_in, project, lambda d: names(d) == ["From first", "From second"])


def test_two_rules_added_at_once_are_both_kept(signed_in: Page, session):
    project = two_rules(signed_in)
    second = session()
    held = [open_live(page, project, hold=True) for page in (signed_in, second)]
    for h in held:
        h.holding = True

    add_rule(signed_in)
    add_rule(second)
    held[0].wait_held(signed_in)
    held[1].wait_held(second)
    for h in held:
        h.release()

    for page in (signed_in, second):
        expect(page.get_by_test_id("rule-card")).to_have_count(4)
    data = stored(signed_in, project, lambda d: len(d["rules"]) == 4)
    assert len({r["rule"]["id"] for r in data["rules"]}) == 4


def test_the_same_field_edited_at_once_converges_to_one_value_everywhere(signed_in: Page, session):
    project = two_rules(signed_in)
    second = session()
    held = [open_live(page, project, hold=True) for page in (signed_in, second)]
    for h in held:
        h.holding = True

    rename_rule(signed_in, "From first")
    rename_rule(second, "From second")
    held[0].wait_held(signed_in)
    held[1].wait_held(second)
    held[0].release()
    held[1].release()

    data = stored(signed_in, project, lambda d: names(d)[0] == "From second")
    for page in (signed_in, second):
        shows(page, names(data))


def test_an_edit_inside_a_rule_another_session_deleted_is_discarded_and_its_author_told(signed_in: Page, session):
    project = two_rules(signed_in)
    second = session()
    open_live(signed_in, project)
    held = open_live(second, project, hold=True)
    held.holding = True
    rename_rule(second, "Doomed edit", rule=1)
    held.wait_held(second)

    signed_in.get_by_test_id("rule-delete").nth(1).click()
    signed_in.get_by_role("button", name="Delete Rule").click()
    expect(second.get_by_test_id("rule-card")).to_have_count(1)
    held.release()

    expect(toast(second, "was discarded")).to_be_visible()
    expect(toast(second, "Doomed edit")).to_be_visible()
    shows(second, ["Rule 1"])
    stored(signed_in, project, lambda d: names(d) == ["Rule 1"])


def test_a_rest_save_reaches_an_open_editor(signed_in: Page):
    project = two_rules(signed_in)
    open_live(signed_in, project)

    data = get_project(signed_in, project)["project_data"]
    data["rules"][0]["meta"]["name"] = "Saved over REST"
    assert signed_in.request.put(f"/api/v1/projects/{project}", data={"project_data": data}).ok

    shows(signed_in, ["Saved over REST", "Rule 2"])


def test_save_waits_for_the_server_and_reports_the_project_saved(signed_in: Page):
    project = two_rules(signed_in)
    open_live(signed_in, project)
    rename_rule(signed_in, "Saved by Save")

    press_save(signed_in)

    expect(toast(signed_in, "Project Saved")).to_be_visible()
    expect(signed_in.get_by_text("Unsaved changes")).to_be_hidden()
    assert names(get_project(signed_in, project)["project_data"]) == ["Saved by Save", "Rule 2"]


# ---- access -------------------------------------------------------------------------------------------------------------

def send_ops(page: Page, project: str) -> dict:
    """Open a live socket from the page, say hello, send one edit; returns the server's answer to it."""
    return page.evaluate("""async ([project, version]) => {
        const ws = new WebSocket(`ws://${location.host}/api/v1/projects/${project}/live`);
        return await new Promise((resolve) => {
            ws.onopen = () => ws.send(JSON.stringify({type: 'hello', since: version, client_id: 'raw'}));
            ws.onmessage = (e) => {
                const m = JSON.parse(e.data);
                if (m.type === 'ready') ws.send(JSON.stringify({type: 'ops', id: 'raw-1', base: m.version,
                    ops: [{op: 'replace', path: '/mode', value: 'all'}]}));
                if (m.msg_id === 'raw-1') resolve(m);
            };
            ws.onclose = (e) => resolve({type: 'closed', code: e.code});
        });
    }""", [project, get_project(page, project)["version"]])


def test_a_read_only_share_sees_edits_live_but_cannot_send(signed_in: Page, other):
    project = two_rules(signed_in)
    share_project(signed_in, project, other.id, "read")
    open_live(signed_in, project)
    open_live(other.page, project)

    rename_rule(signed_in, "Seen by the reader")
    shows(other.page, ["Seen by the reader", "Rule 2"])

    answer = send_ops(other.page, project)
    assert answer["type"] == "rejected" and "read" in answer["reason"]
    rename_rule(other.page, "Reader edit", rule=1)
    expect(toast(other.page, "read-only")).to_be_visible()
    assert names(get_project(signed_in, project)["project_data"]) == ["Seen by the reader", "Rule 2"]
    assert get_project(signed_in, project)["project_data"]["mode"] == "first_match"


def handshake(base_url: str, project: str, cookie: str | None, origin: str | None) -> int:
    """The HTTP status of a websocket handshake on the live route (101 when accepted)."""
    url = urlparse(base_url)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
    headers = {"Connection": "Upgrade", "Upgrade": "websocket", "Sec-WebSocket-Version": "13",
               "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="}
    if cookie:
        headers["Cookie"] = f"session_id={cookie}"
    if origin:
        headers["Origin"] = origin
    conn.request("GET", f"/api/v1/projects/{project}/live", headers=headers)
    status = conn.getresponse().status
    conn.close()
    return status


def test_without_a_session_or_from_another_origin_the_live_socket_refuses(signed_in: Page, base_url: str):
    project = two_rules(signed_in)
    cookie = next(c["value"] for c in signed_in.context.cookies() if c["name"] == "session_id")

    assert handshake(base_url, project, None, base_url) == 403
    assert handshake(base_url, project, cookie, "http://evil.example") == 403
    assert handshake(base_url, project, cookie, base_url) == 101


def test_a_project_i_cannot_read_refuses_the_live_socket(signed_in: Page, other, base_url: str):
    project = two_rules(signed_in)
    cookie = next(c["value"] for c in other.page.context.cookies() if c["name"] == "session_id")

    assert handshake(base_url, project, cookie, base_url) == 403


# ---- restarts and two nodes ----------------------------------------------------------------------------------------------

def restart_backend() -> subprocess.Popen:
    return subprocess.Popen([str(Path(__file__).with_name("stack.sh")), "restart"], stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)


def test_after_a_backend_restart_clients_reconnect_and_an_edit_made_while_it_was_down_is_applied(
        signed_in: Page, session):
    project = two_rules(signed_in)
    second = session()
    for page in (signed_in, second):
        open_live(page, project)

    restarting = restart_backend()
    expect(signed_in.get_by_test_id("live-status")).to_have_text(OFFLINE, timeout=20_000)
    rename_rule(signed_in, "Made while down")
    assert restarting.wait(timeout=240) == 0

    for page in (signed_in, second):
        expect(page.get_by_test_id("live-status")).to_have_text("Live", timeout=30_000)
    expect(toast(signed_in, "Your offline edits were merged")).to_be_visible()
    shows(second, ["Made while down", "Rule 2"])
    stored(signed_in, project, lambda d: names(d) == ["Made while down", "Rule 2"])
    rename_rule(second, "After the restart", rule=1)
    shows(signed_in, ["Made while down", "After the restart"])


@two_nodes
def test_sessions_on_node_one_and_node_two_see_each_others_edits(signed_in: Page, session):
    project = two_rules(signed_in)
    second = session(node=2)
    for page in (signed_in, second):
        open_live(page, project)

    rename_rule(signed_in, "From node one")
    shows(second, ["From node one", "Rule 2"])
    rename_rule(second, "From node two", rule=1)
    shows(signed_in, ["From node one", "From node two"])
    stored(signed_in, project, lambda d: names(d) == ["From node one", "From node two"])


# ---- offline edits ------------------------------------------------------------------------------------------------------

def open_with_line(page: Page, project: str) -> Line:
    line = Line(page)
    open_live(page, project)
    return line


def test_offline_edits_to_another_rule_merge_on_reconnect_and_everyone_sees_them(signed_in: Page, session):
    project = two_rules(signed_in)
    second = session()
    line = open_with_line(signed_in, project)
    open_live(second, project)

    line.cut()
    expect(signed_in.get_by_test_id("live-status")).to_have_text(OFFLINE)
    rename_rule(signed_in, "Made offline", rule=1)
    rename_rule(second, "Made online", rule=0)
    shows(signed_in, ["Rule 1", "Made offline"])
    line.restore()

    expect(toast(signed_in, "Your offline edits were merged")).to_be_visible()
    for page in (signed_in, second):
        shows(page, ["Made online", "Made offline"])
    stored(signed_in, project, lambda d: names(d) == ["Made online", "Made offline"])


def test_an_offline_edit_to_a_field_someone_else_changed_is_refused_and_can_be_downloaded(signed_in: Page, session):
    project = two_rules(signed_in)
    second = session()
    line = open_with_line(signed_in, project)
    open_live(second, project)

    line.cut()
    expect(signed_in.get_by_test_id("live-status")).to_have_text(OFFLINE)
    rename_rule(signed_in, "My offline name")
    rename_rule(second, "Their name")
    stored(second, project, lambda d: names(d)[0] == "Their name")
    line.restore()

    banner = signed_in.get_by_test_id("offline-conflict")
    expect(banner).to_contain_text("My offline name")
    for page in (signed_in, second):
        shows(page, ["Their name", "Rule 2"])
    with signed_in.expect_download() as download:
        banner.get_by_role("button", name="Download my version").click()
    mine = json.loads(Path(download.value.path()).read_text())
    assert names(mine) == ["My offline name", "Rule 2"] and mine["metadata"]["project_id"] == project
    assert names(get_project(signed_in, project)["project_data"]) == ["Their name", "Rule 2"]
    banner.get_by_role("button", name="Dismiss").click()
    expect(banner).to_be_hidden()


def test_one_client_offline_while_two_others_edit_ends_with_everyones_edits(signed_in: Page, session):
    project = two_rules(signed_in)
    second, third = session(), session()
    line = open_with_line(signed_in, project)
    for page in (second, third):
        open_live(page, project)

    line.cut()
    expect(signed_in.get_by_test_id("live-status")).to_have_text(OFFLINE)
    rename_rule(signed_in, "Offline edit", rule=1)
    rename_rule(second, "Second online", rule=0)
    add_rule(third)
    expect(second.get_by_test_id("rule-card")).to_have_count(3)
    line.restore()

    expect(toast(signed_in, "Your offline edits were merged")).to_be_visible()
    for page in (signed_in, second, third):
        shows(page, ["Second online", "Offline edit", "Rule 3"])
    stored(signed_in, project, lambda d: names(d)[:2] == ["Second online", "Offline edit"] and len(d["rules"]) == 3)


def test_a_connection_that_stops_answering_is_noticed_and_the_edits_made_meanwhile_merge(signed_in: Page, session):
    project = two_rules(signed_in)
    second = session()
    line = open_with_line(signed_in, project)
    open_live(second, project)

    line.deaden()
    rename_rule(signed_in, "Into the void", rule=1)
    expect(signed_in.get_by_test_id("live-status")).to_have_text(OFFLINE, timeout=25_000)
    line.restore()

    expect(toast(signed_in, "Your offline edits were merged")).to_be_visible(timeout=15_000)
    shows(second, ["Rule 1", "Into the void"])


@two_nodes
def test_offline_edits_on_node_two_merge_with_edits_made_on_node_one(signed_in: Page, session):
    project = two_rules(signed_in)
    second = session(node=2)
    open_live(signed_in, project)
    line = open_with_line(second, project)

    line.cut()
    expect(second.get_by_test_id("live-status")).to_have_text(OFFLINE)
    rename_rule(signed_in, "On node one", rule=0)
    rename_rule(second, "Offline on node two", rule=1)
    line.restore()

    expect(toast(second, "Your offline edits were merged")).to_be_visible()
    for page in (signed_in, second):
        shows(page, ["On node one", "Offline on node two"])
