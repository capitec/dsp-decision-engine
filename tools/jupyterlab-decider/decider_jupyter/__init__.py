"""Debug decider pipelines in JupyterLab: flows defined in a notebook, or in a Python file."""
from __future__ import annotations

import types

from comm import get_comm_manager

from .serving import TARGET, notebook_comm, serve

__all__ = ["debug"]


def debug(pipeline, data=None, params=None):
    """Open the flow debugger on `pipeline`, to run on `data` with `params`.

    `data` is a polars or pandas DataFrame, or a list of records (default: the notebook's
    `SAMPLE`); `params` a params document (default: the notebook's `PARAMS`). Re-running the
    cell that defines the pipeline is what the debugger's "reload step" picks up.

        from decider_jupyter import debug
        debug(pipeline, df)
    """
    ns = _user_ns()
    name = next((k for k, v in ns.items() if v is pipeline and not k.startswith("_")), "pipeline")

    def module():
        mod = types.ModuleType("__notebook__")
        mod.__dict__.update({name: pipeline, **{k: v for k, v in _user_ns().items() if not k.startswith("__")}})
        if data is not None:
            mod.SAMPLE = data
        if params is not None:
            mod.PARAMS = params
        return mod

    serve(notebook_comm(name), module)


def _user_ns():
    from IPython import get_ipython

    shell = get_ipython()
    return shell.user_ns if shell else {}


def _jupyter_labextension_paths():
    return [{"src": "labextension", "dest": "jupyterlab-decider"}]


# The panel opens a comm to this target to debug a flow file; importing the module is what registers it.
get_comm_manager().register_target(TARGET, lambda comm, _msg: serve(comm))
