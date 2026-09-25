"""The kernel side of the JupyterLab flow debugger: decider's bridge, answering over a comm."""
from __future__ import annotations

import json
import os

from comm import create_comm
from decider_bridge.bridge import Bridge

TARGET = "decider"


def serve(comm, notebook=None):
    """Answer the panel's bridge requests on `comm`: `{id, cmd, ...args}` in, `{id, ok, result | error}` out.

    A request with `fresh` gets a bridge of its own, so whole runs for comparisons and
    scenarios leave the debug session where it was.
    """
    session = Bridge(notebook=notebook)

    def on_msg(msg):
        req = dict(msg["content"]["data"])
        bridge = Bridge(notebook=notebook) if req.pop("fresh", False) else session
        try:
            reply = {"ok": True, "result": os.getcwd() if req["cmd"] == "cwd" else bridge.handle(req)}
        except Exception as e:  # noqa: BLE001  the panel shows the message
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        # Through JSON, so values polars hands back (dates, decimals) arrive as the stdio bridge sends them.
        comm.send(json.loads(json.dumps({"id": req.get("id"), **reply}, default=str)))

    comm.on_msg(on_msg)
    return comm


def notebook_comm(pipeline):
    """A comm that asks the frontend to open a panel on the notebook's `pipeline`."""
    return create_comm(target_name=TARGET, data={"pipeline": pipeline, "cwd": os.getcwd()})
