"""The compiled Arrow import layer: vendored nanoarrow behind a small C shim."""
from decider.engine.boundary._arrow.doctor import available, diagnose

__all__ = ["available", "diagnose"]
