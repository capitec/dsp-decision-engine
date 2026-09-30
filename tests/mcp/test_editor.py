from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest

from decider.mcp import EditorBridge, Policy, build_server, editor_dir, ws_hash
from tests.mcp.test_mcp import _call, _tools


class _Window:
    """A minimal in-process replica of the extension's per-window listener."""

    def __init__(self, workspace: str, sock_dir: Path):
        self.token = "s3cret-token"
        self.actions: list[dict] = []
        self.sock_path = sock_dir / f"{ws_hash(workspace)}.sock"
        self.token_path = sock_dir / f"{ws_hash(workspace)}.token"
        self.token_path.write_text(self.token)
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


def test_editor_tools_are_registered_only_with_a_bridge():
    headless = _tools(build_server(Policy()))
    assert not {"highlight", "reveal_source", "editor_selection"} & set(headless)

    bridged = _tools(build_server(Policy(), EditorBridge(Path("/tmp"))))
    assert {"highlight", "reveal_source", "editor_selection"} <= set(bridged)


def test_editor_tools_are_not_confirmation_gated():
    tools = _tools(build_server(Policy(), EditorBridge(Path("/tmp"))))
    for name in ("highlight", "reveal_source", "editor_selection"):
        assert getattr(tools[name].annotations, "destructive_hint", None) is not True, name


def test_bridge_routes_each_workspace_to_its_own_window(tmp_path):
    win_a = _Window("/w/a", tmp_path)
    win_b = _Window("/w/b", tmp_path)
    mcp = build_server(Policy(), EditorBridge(tmp_path))

    assert _call(mcp, "highlight", workspace="/w/a", flow="credit", nodes=["debt_ratio"])["ok"]
    assert _call(mcp, "highlight", workspace="/w/b", flow="credit", nodes=["approved"])["ok"]
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
