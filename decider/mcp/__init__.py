"""The FastMCP interface over decider core: a `decider mcp` stdio server an agent's MCP host owns.

Headless only: discovery, description, inspection, lineage, source context, run
summaries, check reports, and experiment definitions/results are read tools.
Raw record/trace tools are off unless the `raw` policy opts in; running code or
experiments, starting the debugger and generating ids are confirmation-gated
(destructive). The editor-bound highlight/reveal/selection tools are out of
scope here and land with the VS Code UI (task 11b).
"""
from decider.mcp.policy import REDACTED, Policy
from decider.mcp.server import build_server, run_stdio

__all__ = ["Policy", "REDACTED", "build_server", "run_stdio"]
