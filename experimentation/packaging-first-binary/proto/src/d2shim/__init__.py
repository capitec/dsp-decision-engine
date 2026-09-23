"""d2shim: prototype packaging of decider2's nanoarrow string shim.

    from d2shim import strings
    strings.which()      -> ("compiled" | "pure", reason)
    strings.run_tree(...)
    strings.diagnose()
"""
from .strings import FallbackWarning, diagnose, run_tree, which  # noqa: F401

__all__ = ["FallbackWarning", "diagnose", "run_tree", "which"]
