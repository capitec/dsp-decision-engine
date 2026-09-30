"""The MCP capability policy: which data classes a workspace allows the server to return.

Structural data (flow/step/edge identity, paths, sources) and summarised data
(run tallies, check reports) are always available. Raw record and trace data is
opt-in, off by default, behind the `decider.mcp.rawData` setting; a disabled raw
tool returns the redaction notice below, never a fabricated value.
"""
from __future__ import annotations

REDACTED = "raw data disabled: enable `decider.mcp.rawData` to return it"


class Policy:
    """What a workspace allows the MCP server to read. `raw` is the opt-in."""

    def __init__(self, raw: bool = False):
        self.raw = raw
