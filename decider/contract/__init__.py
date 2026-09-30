"""The static-flow contract: versioned Python models layered over the IR and run layers.

This is what VS Code, notebooks, checks, experiments and MCP all refer to when
they talk about a flow, instead of inventing incompatible identifiers. It is
static structure only: which nodes and value slots a flow has, and how to name
them. Runtime facts — the values a run observed, the params it validated, the
events a session emitted — live on `RunReport` and `Session.events` and are
deliberately out of scope here.

The reference models (`FlowRef`, `StepRef`, `EdgeRef`, `ValueSlotRef`,
`RecordRef`) are the contract owner: `json_schema`, `mcp_schema` and
`typescript` are generated from them, so every consumer shares one schema.

Example::

    from decider.contract import describe
    desc = describe(pipeline)
    for edge in desc.edges:
        print(edge.from_path, edge.to_path, edge.kind)
"""
from decider.contract.describe import Capabilities, capabilities, describe, resolve_step, source_location
from decider.contract.refs import (CONTRACT_VERSION, EdgeRef, FlowDescription, FlowRef, RecordRef, StepRef,
                                   ValueSlotRef, check_version)
from decider.contract.schema import json_schema, mcp_schema, typescript

__all__ = [
    "CONTRACT_VERSION", "Capabilities", "EdgeRef", "FlowDescription", "FlowRef", "RecordRef", "StepRef",
    "ValueSlotRef", "capabilities", "check_version", "describe", "json_schema", "mcp_schema", "resolve_step",
    "source_location", "typescript",
]
