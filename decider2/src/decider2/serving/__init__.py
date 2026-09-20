"""decider2's serving layer — doc 02 §3.6.

    from decider2.serving import app
    asgi_app = app(my_pipeline, mode="live")

Deliberately small and replaceable: `app()` is the whole public surface,
`dispatch.Dispatcher` is the framework-agnostic seam underneath it (doc 02
§3.6: "someone should be able to write their own serving layer pretty
easily"), and nothing outside this package (and `decider2/cli.py`, which
only wires it to a socket) imports it — `graph/`, `compile/`, `params.py`
and `runtime/` all run as a library with no server present.
"""
from __future__ import annotations

from decider2.serving.app import app

__all__ = ["app"]
