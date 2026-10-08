from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest

from decider._tooling.mcp import Policy, build_server
from decider._tooling.mcp._editor import EditorBridge, editor_dir, ws_hash
from tests.mcp.test_mcp import _call, _tools


class _Window:
    """A minimal in-process replica of the extension's per-window listener."""

    def __init__(self, workspace: str, sock_dir: Path):
        self.token = "s3cret-token"
        self.actions: list[dict] = []
        self.sock_path = sock_dir / f"{ws_hash(workspace)}.sock"
        self.token_path = sock_dir / f"{ws_hash(workspace)}.token"
        self.token_path.write_text(self.token)
        (sock_dir / f"{ws_hash(workspace)}.folder").write_text(str(Path(workspace).resolve()))
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(str(self.sock_path))
        self._srv.listen()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            conn, _ = self._srv.accept()
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        try:
            req = json.loads(conn.makefile().readline())
        except (ValueError, OSError):
            conn.close()
            return
        if req.get("token") != self.token:
            conn.sendall(json.dumps({"ok": False, "error": "unauthorized"}).encode() + b"\n")
        else:
            self.actions.append(req["action"])
            result = ({"file": "/w/a/flow.py", "selection": {"text": "x"}}
                      if req["action"]["kind"] == "selection" else {"ack": True})
            conn.sendall(json.dumps({"ok": True, "result": result}).encode() + b"\n")
        conn.close()


READ_ONLY = ("highlight", "reveal_source", "editor_selection", "show_flow", "set_breakpoints")
DEBUGGING = ("debug_flow", "debug_control", "debug_evaluate")


def test_editor_tools_are_registered_only_with_a_bridge():
    headless = _tools(build_server(Policy()))
    assert not {*READ_ONLY, *DEBUGGING} & set(headless)

    bridged = _tools(build_server(Policy(), EditorBridge(Path("/tmp"))))
    assert {*READ_ONLY, *DEBUGGING} <= set(bridged)


def test_only_the_debugging_tools_are_confirmation_gated():
    tools = _tools(build_server(Policy(), EditorBridge(Path("/tmp"))))
    for name in READ_ONLY:
        assert getattr(tools[name].annotations, "destructive_hint", None) is not True, name
    for name in DEBUGGING:
        assert tools[name].annotations.destructive_hint is True, name


def test_bridge_routes_each_workspace_to_its_own_window(tmp_path):
    win_a = _Window("/w/a", tmp_path)
    win_b = _Window("/w/b", tmp_path)
    mcp = build_server(Policy(), EditorBridge(tmp_path))

    assert _call(mcp, "highlight", workspace="/w/a", nodes=["debt_ratio"])["ok"]
    assert _call(mcp, "highlight", workspace="/w/b", nodes=["approved"])["ok"]
    assert [a["nodes"] for a in win_a.actions] == [["debt_ratio"]]
    assert [a["nodes"] for a in win_b.actions] == [["approved"]]


def test_bridge_returns_the_editors_selection(tmp_path):
    _Window("/w/a", tmp_path)
    mcp = build_server(Policy(), EditorBridge(tmp_path))
    out = _call(mcp, "editor_selection", workspace="/w/a")
    assert out["ok"] is True
    assert out["result"]["selection"]["text"] == "x"


def test_bridge_rejects_a_wrong_token(tmp_path):
    win = _Window("/w/a", tmp_path)
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.connect(str(win.sock_path))
    c.sendall(json.dumps({"token": "wrong", "action": {"kind": "highlight"}}).encode() + b"\n")
    assert json.loads(c.makefile().readline())["error"] == "unauthorized"
    c.close()


def test_bridge_reports_a_workspace_with_no_window(tmp_path):
    out = EditorBridge(tmp_path).send("/w/nowhere", {"kind": "highlight"})
    assert out["ok"] is False
    assert "no editor window for workspace '/w/nowhere'" in out["error"]


def test_ws_hash_is_stable_and_resolves_symlinks(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    assert ws_hash(str(link)) == ws_hash(str(real))


def test_editor_dir_is_owner_only(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    d = editor_dir()
    assert d == tmp_path / ".decider" / "editor"
    assert (d.stat().st_mode & 0o777) == 0o700


@pytest.mark.parametrize("workspace", ["/w/a", "relative/workspace"])
def test_ws_hash_is_absolute_and_deterministic(workspace):
    assert ws_hash(workspace) == ws_hash(workspace)
    assert len(ws_hash(workspace)) == 16


def _folder(tmp_path, *parts):
    d = tmp_path.joinpath("ws", *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_workspace_resolves_to_the_nearest_ancestor_window(tmp_path):
    outer, inner = _folder(tmp_path, "proj"), _folder(tmp_path, "proj", "sub")
    win_outer, win_inner = _Window(str(outer), tmp_path), _Window(str(inner), tmp_path)
    deep = _folder(tmp_path, "proj", "sub", "configs", "0.1.0")
    EditorBridge(tmp_path).send(str(deep), {"kind": "selection"})
    assert len(win_inner.actions) == 1 and not win_outer.actions
    EditorBridge(tmp_path).send(str(_folder(tmp_path, "proj", "other")), {"kind": "selection"})
    assert len(win_outer.actions) == 1


def test_workspace_defaults_to_the_servers_working_directory(tmp_path, monkeypatch):
    proj = _folder(tmp_path, "proj")
    win = _Window(str(proj), tmp_path)
    monkeypatch.chdir(proj)
    mcp = build_server(Policy(), EditorBridge(tmp_path))
    assert _call(mcp, "editor_selection")["ok"] is True
    assert len(win.actions) == 1


def test_workspace_above_exactly_one_window_uses_it(tmp_path):
    win = _Window(str(_folder(tmp_path, "mono", "svc")), tmp_path)
    out = EditorBridge(tmp_path).send(str(_folder(tmp_path, "mono")), {"kind": "selection"})
    assert out["ok"] is True and len(win.actions) == 1


def test_workspace_above_several_windows_asks_which(tmp_path):
    for name in ("a", "b"):
        _Window(str(_folder(tmp_path, "mono", name)), tmp_path)
    out = EditorBridge(tmp_path).send(str(_folder(tmp_path, "mono")), {"kind": "selection"})
    assert out["ok"] is False
    assert "2 VS Code windows" in out["error"] and "mono/a" in out["error"] and "mono/b" in out["error"]


def test_no_window_says_what_was_tried_and_what_to_do(tmp_path):
    out = EditorBridge(tmp_path).send(str(_folder(tmp_path, "empty")), {"kind": "selection"})
    assert out["ok"] is False
    assert "tried" in out["error"] and str(tmp_path / "ws" / "empty") in out["error"]
    assert "extension must be running in a window with that folder open" in out["error"]


def test_a_dead_socket_is_reported_as_such(tmp_path):
    proj = _folder(tmp_path, "proj")
    (tmp_path / f"{ws_hash(str(proj))}.sock").write_text("")
    (tmp_path / f"{ws_hash(str(proj))}.token").write_text("t")
    out = EditorBridge(tmp_path).send(str(proj), {"kind": "selection"})
    assert out["ok"] is False
    assert "dead socket" in out["error"] and str(proj) in out["error"]


def test_a_dead_ancestor_does_not_hide_a_live_window(tmp_path):
    _Window(str(_folder(tmp_path, "mono", "svc")), tmp_path)
    mono = _folder(tmp_path, "mono")
    (tmp_path / f"{ws_hash(str(mono))}.sock").write_text("")
    out = EditorBridge(tmp_path).send(str(mono), {"kind": "selection"})
    assert out["ok"] is True


def test_navigation_tools_send_absolute_paths_and_arguments(tmp_path):
    proj = _folder(tmp_path, "proj")
    win = _Window(str(proj), tmp_path)
    mcp = build_server(Policy(), EditorBridge(tmp_path))
    ws = str(proj)
    _call(mcp, "show_flow", workspace=ws, file="flows/main.yaml", pipeline="main")
    _call(mcp, "reveal_source", workspace=ws, path="steps.py", line=7)
    _call(mcp, "set_breakpoints", workspace=ws, nodes=["credit/limit"])
    _call(mcp, "debug_flow", workspace=ws, file="flows/main.yaml", dataset="applicants")
    _call(mcp, "debug_control", workspace=ws, action="step_over")
    assert win.actions == [
        {"kind": "show_flow", "file": str(proj / "flows/main.yaml"), "pipeline": "main"},
        {"kind": "reveal", "path": str(proj / "steps.py"), "line": 7},
        {"kind": "set_breakpoints", "nodes": ["credit/limit"]},
        {"kind": "debug_flow", "file": str(proj / "flows/main.yaml"), "pipeline": None, "dataset": "applicants"},
        {"kind": "debug_control", "action": "step_over"},
    ]


def test_debug_evaluate_is_redacted_unless_raw_is_enabled(tmp_path):
    proj = _folder(tmp_path, "proj")
    win = _Window(str(proj), tmp_path)
    off = build_server(Policy(), EditorBridge(tmp_path))
    assert "redacted" in _call(off, "debug_evaluate", workspace=str(proj), expression="income")
    assert win.actions == []

    on = build_server(Policy(raw=True), EditorBridge(tmp_path))
    assert _call(on, "debug_evaluate", workspace=str(proj), expression="income")["ok"] is True
    assert win.actions == [{"kind": "debug_evaluate", "expression": "income"}]


@pytest.mark.parametrize("env, raw", [({"DECIDER_MCP_RAW": "1"}, True), ({}, False)])
def test_mcp_command_reads_raw_from_the_environment(monkeypatch, env, raw):
    from click.testing import CliRunner

    from decider._tooling import mcp as mcp_pkg
    from decider._tooling.cli import cli

    seen = []
    monkeypatch.setattr(mcp_pkg, "run_stdio", lambda raw: seen.append(raw))
    monkeypatch.delenv("DECIDER_MCP_RAW", raising=False)
    assert CliRunner().invoke(cli, ["mcp"], env=env).exit_code == 0
    assert seen == [raw]
