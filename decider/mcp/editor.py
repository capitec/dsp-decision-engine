"""The MCP process's client to a VS Code window: the authenticated editor bridge.

Editor-bound tools (highlight, reveal, selection) are not answered in the MCP
process; they are forwarded to the VS Code window that owns a workspace, over a
per-window Unix-domain socket (a named pipe on Windows) authenticated by a
per-window secret token stored beside it in an owner-only directory.

Threat note: this is where broad MCP read scope meets a live editor, so the
channel is authenticated even between processes of the same user — the socket's
0600 ACL keeps out other users, the token keeps out other local processes.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path


def editor_dir() -> Path:
    """The machine-local, owner-only directory holding every window's socket and token."""
    d = Path.home() / ".decider" / "editor"
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o700)
    return d


def ws_hash(workspace: str) -> str:
    """The socket/token filename for a workspace: a stable hash of its real path."""
    return hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:16]


class EditorBridge:
    """Forward an editor action to the VS Code window that owns `workspace`.

    Connects to the window's socket, presents the token read from the token
    file, and returns the window's reply. A missing window, a wrong token, or an
    unreachable socket is reported in the reply dict, never raised, so an
    editor-bound tool can return the failure plainly to the agent.
    """

    def __init__(self, sock_dir: Path | None = None):
        self.sock_dir = sock_dir or editor_dir()

    def send(self, workspace: str, action: dict) -> dict:
        name = ws_hash(workspace)
        sock = self.sock_dir / f"{name}.sock"
        token = self.sock_dir / f"{name}.token"
        if not sock.exists() or not token.exists():
            return {"ok": False, "error": f"no editor window for workspace {workspace!r}"}
        req = json.dumps({"token": token.read_text(), "action": action}).encode() + b"\n"
        try:
            return _roundtrip(str(sock), req)
        except (OSError, ValueError) as e:
            return {"ok": False, "error": f"editor window unreachable: {e}"}


def _roundtrip(sock_path: str, req: bytes) -> dict:
    if os.name == "nt":
        # Windows has no AF_UNIX; the extension's `net` server names the pipe by
        # the socket path, so the pipe is opened as a file with the same handshake.
        # ponytail: untested here (no Windows runner); the Unix path is the tested one.
        with open("\\\\.\\pipe\\" + sock_path.replace("/", "\\"), "r+b") as pipe:
            pipe.write(req)
            pipe.flush()
            return json.loads(pipe.readline())
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(sock_path)
        s.sendall(req)
        return json.loads(s.makefile().readline())
