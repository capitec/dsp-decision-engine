"""Offline edits in the live flat-rule editor: made while a browser's socket is cut or dead, merged on reconnect or
refused whole with a banner. Runs only on a stack started with DSP_LIVE_COLLAB=true, like `test_live_collab.py`.
"""

import json
import os
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from platform_ui import toast
from seed import get_project
from test_live_collab import (LIVE, OFFLINE, add_rule, names, open_live, session, shows, stored,  # noqa: F401
                              two_nodes, two_rules)
from flat_rules_ui import rename_rule

pytestmark = pytest.mark.skipif(os.environ.get("E2E_LIVE_COLLAB") != "true",
                                reason="needs a stack started with DSP_LIVE_COLLAB=true")


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


class Padded:
    """Pads the `ops` messages a page sends on its live socket past the server's 4 MiB limit while `padding`, so the
    server closes the socket on them every time."""

    def __init__(self, page: Page):
        self.padding = False
        page.route_web_socket(LIVE, self._route)

    def _route(self, ws):
        server = ws.connect_to_server()

        def outgoing(message):
            data = json.loads(message)
            if self.padding and data.get("type") == "ops":
                message = json.dumps({**data, "pad": "x" * (4 * 1024 * 1024 + 1)})
            server.send(message)

        ws.on_message(outgoing)


def test_edits_the_server_fails_on_every_time_are_handed_back_after_three_tries(signed_in: Page):
    project = two_rules(signed_in)
    padded = Padded(signed_in)
    open_live(signed_in, project)

    padded.padding = True
    rename_rule(signed_in, "Never stored")

    banner = signed_in.get_by_test_id("offline-conflict")
    expect(banner).to_contain_text("3 times", timeout=30_000)
    shows(signed_in, ["Rule 1", "Rule 2"])
    with signed_in.expect_download() as download:
        banner.get_by_role("button", name="Download my version").click()
    assert names(json.loads(Path(download.value.path()).read_text())) == ["Never stored", "Rule 2"]
    padded.padding = False
    rename_rule(signed_in, "Stored", rule=1)
    stored(signed_in, project, lambda d: names(d) == ["Rule 1", "Stored"])
