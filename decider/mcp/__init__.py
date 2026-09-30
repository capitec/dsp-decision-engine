"""The FastMCP interface over decider core: a `decider mcp` stdio server an agent's MCP host owns.

Headless tools — discovery, description, inspection, lineage, source context,
run summaries, check reports, and experiment definitions/results — answer from
core alone. Raw record/trace tools are off unless the `raw` policy opts in;
running code or experiments, starting the debugger and generating ids are
confirmation-gated (destructive). The editor-bound highlight/reveal/selection
tools are only registered when an `EditorBridge` is given, and forward to the
VS Code window that owns a workspace over an authenticated per-window socket.
"""
from decider.mcp.editor import EditorBridge, editor_dir, ws_hash
from decider.mcp.policy import REDACTED, Policy
from decider.mcp.server import build_server, run_stdio

__all__ = ["Policy", "REDACTED", "EditorBridge", "editor_dir", "ws_hash", "build_server", "run_stdio"]
