"""The compiled Arrow import layer: vendored nanoarrow behind a small C shim.

`available()` and `diagnose()` work without the shim. `plan` and `intrinsics`
don't need it either; `view` and `kernels` do, and raise one `ImportError`
when it can't be built.
"""
from decider.engine.boundary._arrow.doctor import available, diagnose

__all__ = ["available", "diagnose"]
