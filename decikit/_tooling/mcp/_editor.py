"""The MCP process's client to a VS Code window: the authenticated editor bridge.

Editor-bound tools (highlight, reveal, selection) are not answered in the MCP
process; they are forwarded to the VS Code window that owns a workspace, over a
per-window Unix-domain socket (a named pipe on Windows) authenticated by a
per-window secret token stored beside it in an owner-only directory. A `.folder`
file beside them names the window's folder, so a workspace above it can find it.

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
    d = Path.home() / ".decikit" / "editor"
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o700)
    return d


def ws_hash(workspace: str) -> str:
    """The socket/token filename for a workspace: a stable hash of its real path."""
    return hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:16]


class EditorBridge:
    """Forward an editor action to the VS Code window that owns `workspace`.

    The window is the one whose folder is `workspace` or its nearest ancestor;
    failing that, the only live window whose folder lies under `workspace`. It
    presents the token read from the token file and returns the window's reply.
    A missing window, a wrong token, or an unreachable socket is reported in the
    reply dict, never raised, so an editor-bound tool can return the failure
    plainly to the agent.
    """

    def __init__(self, sock_dir: Path | None = None):
        self.sock_dir = sock_dir or editor_dir()

    def send(self, workspace: str, action: dict) -> dict:
        here = Path(workspace).resolve()
        tried = [here, *here.parents]
        stale = [p for p in tried if self._sock(p).exists() and not _alive(self._sock(p))]
        live = [p for p in tried if self._sock(p).exists() and p not in stale]
        folder = live[0] if live else self._only_below(here)
        if isinstance(folder, str):
            return {"ok": False, "error": folder}
        if folder is None:
            return {"ok": False, "error": self._missing(workspace, tried, stale)}
        token = self._sock(folder).with_suffix(".token")
        if not token.exists():
            return {"ok": False, "error": f"editor window for {str(folder)!r} has no token file; reload its VS Code window"}
        req = json.dumps({"token": token.read_text(), "action": action}).encode() + b"\n"
        try:
            return _roundtrip(str(self._sock(folder)), req)
        except (OSError, ValueError) as e:
            return {"ok": False, "error": f"editor window unreachable: {e}"}

    def _sock(self, folder: Path) -> Path:
        return self.sock_dir / f"{ws_hash(str(folder))}.sock"

    def _only_below(self, here: Path) -> Path | str | None:
        """The one live window whose folder is under `here`; a message when several are."""
        below = []
        for f in sorted(self.sock_dir.glob("*.folder")):
            folder = Path(f.read_text().strip())
            if folder.is_relative_to(here) and _alive(f.with_suffix(".sock")):
                below.append(folder)
        if len(below) > 1:
            return f"{len(below)} VS Code windows are open under {str(here)!r}: pass workspace= one of {[str(p) for p in below]}"
        return below[0] if below else None

    def _missing(self, workspace: str, tried: list[Path], stale: list[Path]) -> str:
        msg = (f"no editor window for workspace {workspace!r}: tried {', '.join(map(str, tried))}, "
               "and no single open window below it. The decikit VS Code extension must be running "
               "in a window with that folder open.")
        if stale:
            msg += (f" The window for {', '.join(map(str, stale))} left a dead socket "
                    "(it crashed or closed); reopen the folder in VS Code.")
        return msg


def _alive(sock: Path) -> bool:
    if os.name == "nt":
        return sock.exists()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        try:
            s.connect(str(sock))
        except OSError:
            return False
    return True


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
        s.settimeout(120)  # debug_flow and debug_control wait for the next pause
        s.connect(sock_path)
        s.sendall(req)
        return json.loads(s.makefile().readline())
