"""The decider debug bridge: the Python side of the VS Code and JupyterLab debuggers.

This package is the editor-facing transport and adapter over
`decider.engine.debug.Session`. It owns the stdio JSON-lines protocol, the
debugpy attach flag and the process launch; it does not implement a second
session. Session semantics live in `decider.engine.debug`.
"""
