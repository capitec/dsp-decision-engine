"""Prototype for experiment 04: FastMCP topology and editor bridge.

Answers the four spike questions with the smallest runnable thing:

1. MCP process ownership/lifecycle/transport -> a `decider`-hosted FastMCP
   process owned by the agent's MCP host, spawned as `uv run decider mcp`, over
   stdio. The server is stateless per session: each tool call discovers/loads
   what it needs (exactly the pattern `Bridge.describe`/`start` use today).
2. Editor-bridge auth + window routing -> one Unix-domain socket per VS Code
   window plus a per-window secret token in a machine-local, owner-only
   directory. The workspace folder resolves the socket; the token authenticates
   the caller; a wrong or missing token is refused.
3. Headless vs editor-bound split -> headless tools answer from `decider` core
   alone; editor-bound tools forward an action to the resolved window and return
   an ack, never a round-trip of the editor's own state.
4. Capability policy -> structural (always), summarised (default on), raw
   (opt-in, default off). Disabled raw tools return a redaction notice, never
   fabricated data.

A minimal fake core stands in for the real modules so the spike runs without
numba compilation; task 11 binds the headless tools to `engine/ir` (`to_ir`,
`step_map`, `Origin`), `engine/run` (`RunReport`), `testing/`, and the debug
bridge, and the editor-bound tools to the extension's real listener.

Run:
    uv run --with fastmcp python notes/vscode-redesign/experimentation/04-fastmcp/prototype.py
Serve over stdio (for an agent's MCP config):
    uv run --with fastmcp python notes/vscode-redesign/experimentation/04-fastmcp/prototype.py --serve
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from fastmcp import FastMCP

REDACTED = "raw data disabled: enable `decider.mcp.rawData` to return it"


# ---- capability policy (question 4) -----------------------------------------

class Policy:
    """What a workspace allows the MCP server to read.

    `raw` is the opt-in; structural and summarised are always available.
    """

    def __init__(self, raw: bool = False):
        self.raw = raw


# ---- fake core: the headless capabilities task 11 binds to real modules ------

def _flows(root: str) -> list[dict]:
    return [{"name": "credit", "source": f"{root}/credit/pipeline.py",
             "steps": ["debt_ratio", "approved"]}]


def _describe(flow: str) -> dict:
    # Real binding: to_ir + step_map -> the structural flow description.
    return {"flow": flow, "steps": [
        {"id": "3f9a7c2e1b5d", "path": "debt_ratio", "reads": ["income", "debt"],
         "writes": ["debt_ratio"], "source": "credit/pipeline.py:4"},
        {"id": "8b1d4f00a5c3", "path": "approved", "reads": ["debt_ratio"],
         "writes": ["approved"], "source": "credit/pipeline.py:8"},
    ]}


def _run_summary(flow: str) -> dict:
    # Real binding: RunReport -> counts, no per-record values.
    return {"flow": flow, "rows": 1000, "approved": 412, "declined": 588}


def _raw_record(flow: str, record_id: str) -> dict:
    # Real binding: session/run state for one record -> values per step.
    return {"flow": flow, "record_id": record_id,
            "values": {"debt_ratio": 0.31, "approved": True}}


def _raw_trace(flow: str, record_id: str) -> dict:
    # Real binding: the decision-trace events for one record.
    return {"flow": flow, "record_id": record_id,
            "events": [{"step": "debt_ratio", "when": "after", "value": 0.31}]}


# ---- editor bridge (question 2) ---------------------------------------------

def _ws_hash(workspace: str) -> str:
    return hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:16]


class EditorWindow:
    """The extension-side listener for one VS Code window (mock in the spike).

    Binds a Unix socket named by the workspace and writes a fresh random token;
    both files are mode 0600 so only the owning user can read/connect.
    """

    def __init__(self, workspace: str, sock_dir: Path):
        self.workspace = str(Path(workspace).resolve())
        self.token = secrets.token_hex(32)
        self.sock_path = sock_dir / f"{_ws_hash(workspace)}.sock"
        self.token_path = sock_dir / f"{_ws_hash(workspace)}.token"
        self.actions: list[dict] = []
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(str(self.sock_path))
        self._srv.listen()
        self.token_path.write_text(self.token)
        self.token_path.chmod(0o600)
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
            conn.sendall(json.dumps({"ok": False, "error": "unauthorized"}).encode())
        else:
            self.actions.append(req["action"])
            conn.sendall(json.dumps({"ok": True, "ack": req["action"]}).encode())
        conn.close()


class EditorBridge:
    """The MCP-process side: route an action to the window for a workspace."""

    def __init__(self, sock_dir: Path):
        self.sock_dir = sock_dir

    def send(self, workspace: str, action: dict) -> dict:
        sock = self.sock_dir / f"{_ws_hash(workspace)}.sock"
        token = self.sock_dir / f"{_ws_hash(workspace)}.token"
        if not sock.exists() or not token.exists():
            return {"ok": False, "error": f"no editor window for workspace {workspace!r}"}
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.connect(str(sock))
        c.sendall(json.dumps({"token": token.read_text(), "action": action}).encode() + b"\n")
        reply = json.loads(c.makefile().readline())
        c.close()
        return reply


# ---- the MCP server (questions 1 and 3) --------------------------------------

def build_server(policy: Policy, bridge: EditorBridge) -> FastMCP:
    mcp = FastMCP(name="decider")

    # Headless, structural (always available).
    @mcp.tool
    def discover_flows() -> list[dict]:
        """List the decider pipelines in the workspace and their source files."""
        return _flows(os.getcwd())

    @mcp.tool
    def describe_flow(flow: str) -> dict:
        """The structural description of a flow: steps, reads, writes, sources."""
        return _describe(flow)

    # Headless, summarised (always available).
    @mcp.tool
    def run_summary(flow: str) -> dict:
        """Row counts and outcome tallies for a flow's latest run."""
        return _run_summary(flow)

    # Headless, raw (gated on `decider.mcp.rawData`).
    @mcp.tool
    def raw_record(flow: str, record_id: str) -> dict:
        """The per-step values for one record (raw data, opt-in)."""
        if not policy.raw:
            return {"redacted": REDACTED}
        return _raw_record(flow, record_id)

    @mcp.tool
    def raw_trace(flow: str, record_id: str) -> dict:
        """The decision-trace events for one record (raw data, opt-in)."""
        if not policy.raw:
            return {"redacted": REDACTED}
        return _raw_trace(flow, record_id)

    # Editor-bound: forward to the resolved window, return an ack.
    @mcp.tool
    def highlight(workspace: str, flow: str, nodes: list[str]) -> dict:
        """Ask the editor to highlight flow entities in the window for a workspace."""
        return bridge.send(workspace, {"kind": "highlight", "flow": flow, "nodes": nodes})

    @mcp.tool
    def reveal_source(workspace: str, path: str, line: int) -> dict:
        """Ask the editor to reveal a source range in the window for a workspace."""
        return bridge.send(workspace, {"kind": "reveal", "path": path, "line": line})

    @mcp.tool
    def editor_selection(workspace: str) -> dict:
        """Ask the editor for its current selection (editor-bound read)."""
        return bridge.send(workspace, {"kind": "selection"})

    # Confirmation-gated: runs code, marked destructive so the client gates it.
    @mcp.tool(annotations={"destructiveHint": True})
    def run_flow(flow: str) -> dict:
        """Run a flow end to end. Requires client confirmation before it runs."""
        return {"flow": flow, "ran": True}

    return mcp


# ---- checks ------------------------------------------------------------------

def _call(mcp: FastMCP, name: str, **args):
    r = asyncio.run(mcp.call_tool(name, args))
    assert not r.is_error, f"{name}: {r}"
    return r.structured_content


def _stdio_roundtrip() -> None:
    """Start a real stdio server subprocess and drive the MCP handshake."""
    proc = subprocess.Popen(
        [sys.executable, __file__, "--serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    send = lambda msg: (proc.stdin.write(json.dumps(msg) + "\n"), proc.stdin.flush())
    recv = lambda mid: next(
        json.loads(l) for l in iter(proc.stdout.readline, "")
        if json.loads(l).get("id") == mid
    )
    send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
          "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                     "clientInfo": {"name": "spike", "version": "0.0.0"}}})
    init = recv(1)
    assert "serverInfo" in init["result"], init
    send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = {t["name"] for t in recv(2)["result"]["tools"]}
    assert {"discover_flows", "describe_flow", "highlight", "raw_record"} <= tools, tools
    send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
          "params": {"name": "describe_flow", "arguments": {"flow": "credit"}}})
    reply = recv(3)
    assert reply["result"]["isError"] is False, reply
    send({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
          "params": {"name": "raw_record", "arguments": {"flow": "credit", "record_id": "r1"}}})
    # Raw is disabled by default: the stdio server answers redacted.
    assert recv(4)["result"]["structuredContent"]["redacted"] == REDACTED
    proc.terminate()
    proc.wait(timeout=10)  # a clean lifecycle: terminate ends the process, no hang


def _sock_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="decider-mcp-"))


def demo() -> None:
    sock_dir = _sock_dir()
    bridge = EditorBridge(sock_dir)

    # 1. capability discovery: every tool is listed, headless and editor-bound.
    mcp = build_server(Policy(raw=False), bridge)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"discover_flows", "describe_flow", "run_summary", "raw_record", "raw_trace",
            "highlight", "reveal_source", "editor_selection", "run_flow"} <= names, names

    # 2. structural and summarised are always available.
    assert _call(mcp, "describe_flow", flow="credit")["flow"] == "credit"
    assert _call(mcp, "run_summary", flow="credit")["rows"] == 1000

    # 3. raw is gated: off by default, on when the policy opts in.
    assert _call(mcp, "raw_record", flow="credit", record_id="r1")["redacted"] == REDACTED
    mcp_raw = build_server(Policy(raw=True), bridge)
    assert _call(mcp_raw, "raw_record", flow="credit", record_id="r1")["values"]["approved"] is True

    # 4. editor-bound routing: two windows, each gets only its own action.
    win_a = EditorWindow("/w/a", sock_dir)
    win_b = EditorWindow("/w/b", sock_dir)
    assert _call(mcp, "highlight", workspace="/w/a", flow="credit", nodes=["debt_ratio"])["ok"]
    assert _call(mcp, "highlight", workspace="/w/b", flow="credit", nodes=["approved"])["ok"]
    assert [a["nodes"] for a in win_a.actions] == [["debt_ratio"]]
    assert [a["nodes"] for a in win_b.actions] == [["approved"]]

    # 5. auth: a caller with the wrong token is refused; a workspace with no
    #    window is refused with a routing error.
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.connect(str(win_a.sock_path))
    c.sendall(json.dumps({"token": "wrong", "action": {"kind": "highlight"}}).encode() + b"\n")
    assert json.loads(c.makefile().readline())["error"] == "unauthorized"
    c.close()
    assert bridge.send("/w/nowhere", {"kind": "highlight"})["ok"] is False
    assert "no editor window" in bridge.send("/w/nowhere", {"kind": "highlight"})["error"]

    # 6. confirmation gate is discoverable as a destructive annotation.
    tool = next(t for t in asyncio.run(mcp.list_tools()) if t.name == "run_flow")
    assert tool.annotations.destructive_hint is True

    # 7. transport + lifecycle: a real stdio subprocess serves the protocol and
    #    exits cleanly on terminate.
    _stdio_roundtrip()

    print("prototype ok")


if __name__ == "__main__":
    if "--serve" in sys.argv:
        build_server(Policy(raw=False), EditorBridge(_sock_dir())).run(transport="stdio")
    else:
        demo()
