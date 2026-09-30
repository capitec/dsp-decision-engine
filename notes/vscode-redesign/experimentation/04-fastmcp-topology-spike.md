# Experiment 04 — FastMCP topology and editor bridge

**Run during:** task 11, before MCP transport and lifecycle implementation freezes  
**Feeds:** task 11, task 12

## Question

How can a Python FastMCP server serve headless `decider` capabilities and
securely request editor-bound actions from the correct VS Code window?

## Method

- Prototype a `decider`-hosted FastMCP process using stdio for agent transport.
- Separate headless core tools from extension-bound highlight/reveal tools.
- Prototype the minimum authenticated local IPC required between the extension
  and the MCP process; do not assume an unauthenticated localhost listener.
- Test workspace scoping, process lifecycle, capability discovery, raw-data
  availability, capability controls for structural/summarised/raw data,
  confirmation gates, error reporting, and multiple-window behaviour.

## Decision outputs

- MCP process ownership, lifecycle, and transport.
- Editor-bridge authentication and workspace/window routing.
- The headless/editor-bound tool split.
- Capability policy for structural data, summaries, raw records, and raw trace
  evidence.
