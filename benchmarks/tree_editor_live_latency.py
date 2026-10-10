"""Edit-to-other-browser latency of live editing in the flat-rule editor, on one node and across two.

Browser A renames rule 1 N times; browser B (on node 1, then on node 2) records when its rule list shows each name.
Both pages run in one Chromium on one machine, so `Date.now()` is one clock. Printed per setup: click -> shown in B
(what a user sees), click -> socket send (the debounce, plus diffing) and send -> shown in B (server and fan-out).

    BASE=http://localhost:8710 BASE_2=http://localhost:8711 N=50 uv run --no-project \
        --with-requirements ui/tree-editor/e2e/requirements.txt --with-editable . python benchmarks/tree_editor_live_latency.py
"""

import os
import statistics
import sys
import uuid
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui/tree-editor/e2e"))
from flat_docs import SMALL, document  # noqa: E402
from tree_ui import register, sign_in  # noqa: E402

BASE, BASE_2 = os.environ.get("BASE", "http://localhost:8710"), os.environ.get("BASE_2")
N = int(os.environ.get("N", "50"))

SENDER = """
window.__clicks = []; window.__sends = [];
document.addEventListener('click', (e) => {
  if (e.target.closest('[data-testid=rule-rename-done]')) window.__clicks.push(Date.now());
}, true);
const send = WebSocket.prototype.send;
WebSocket.prototype.send = function (data) {
  if (String(data).includes('"type":"ops"')) window.__sends.push(Date.now());
  return send.call(this, data);
};
"""
RECEIVER = """
window.__seen = {};
new MutationObserver(() => {
  const name = document.querySelector('[data-testid=rule-name]')?.textContent;
  if (name && !(name in window.__seen)) window.__seen[name] = Date.now();
}).observe(document, {subtree: true, childList: true, characterData: true});
"""


def pct(values, p):
    return sorted(values)[min(len(values) - 1, int(len(values) * p))]


def summary(label, values):
    return f"{label} p50 {statistics.median(values):.0f} p95 {pct(values, 0.95):.0f} ms"


def measure(browser, email, receiver_base, label):
    a = browser.new_context(base_url=BASE, viewport={"width": 1500, "height": 950}).new_page()
    a.add_init_script(SENDER)
    a.goto("/")
    sign_in(a, email)
    expect(a.get_by_role("button", name="Overview")).to_be_visible()
    r = a.request.post("/api/v1/projects", data={"name": "Latency", "project_type": "flat_rule",
                                                 "project_data": document(SMALL, SMALL)})
    project = r.json()["id"]
    b = browser.new_context(base_url=receiver_base, viewport={"width": 1500, "height": 950}).new_page()
    b.add_init_script(RECEIVER)
    b.goto("/")
    sign_in(b, email)
    expect(b.get_by_role("button", name="Overview")).to_be_visible()
    for page in (a, b):
        page.goto(f"/flat-rules/{project}")
        expect(page.get_by_test_id("live-status")).to_have_text("Live", timeout=30_000)
    card = a.get_by_test_id("rule-card").first
    total, debounce, server = [], [], []
    for i in range(N):
        name = f"Latency {i}"
        card.get_by_test_id("rule-rename").click()
        card.get_by_role("textbox").first.fill(name)
        card.get_by_test_id("rule-rename-done").click()
        expect(b.get_by_test_id("rule-name").first).to_have_text(name)
        clicked, sent = a.evaluate("[window.__clicks.at(-1), window.__sends.at(-1)]")
        seen = b.evaluate(f"window.__seen[{name!r}]")
        total.append(seen - clicked)
        debounce.append(sent - clicked)
        server.append(seen - sent)
    print(f"{label} (n={N}, project {project}): {summary('click->shown', total)}; "
          f"{summary('click->sent', debounce)}; {summary('sent->shown', server)}")
    a.context.close()
    b.context.close()


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(base_url=BASE)
    email = f"{uuid.uuid4().hex[:10]}@example.com"
    page.goto(BASE)
    register(page, email)
    page.close()
    measure(browser, email, BASE, "one node")
    if BASE_2:
        measure(browser, email, BASE_2, "two nodes")
    browser.close()
