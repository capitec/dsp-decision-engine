"""Backend selection: compiled (nanoarrow shim) when the extension imports,
pure numba otherwise. This is the install-time behaviour the package
actually has on a platform with no wheel.

Policy, in one place:

    D2SHIM_BACKEND unset       -> compiled if importable, else pure + ONE RuntimeWarning
    D2SHIM_BACKEND=compiled    -> compiled, or raise ImportError with the real cause (production)
    D2SHIM_BACKEND=pure        -> pure, never touch the extension (also how the fallback is tested)

The decision is made once, at first use, and is inspectable through
`which()` and `diagnose()`.
"""
from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
import warnings
from functools import lru_cache

from . import _arrowc as arrowc
from . import pure

ENV = "D2SHIM_BACKEND"


class FallbackWarning(RuntimeWarning):
    """Raised (once) when the compiled backend was wanted but is unavailable."""


@lru_cache(maxsize=None)
def _probe():
    """Try the extension exactly once. Returns (module_or_None, ImportError_or_None)."""
    try:
        from . import compiled
        return compiled, None
    except ImportError as e:  # covers: no wheel for this platform, build skipped, ABI mismatch
        return None, e


@lru_cache(maxsize=None)
def which() -> tuple[str, str]:
    """(backend name, one-line reason). Decided once per process."""
    want = os.environ.get(ENV, "").strip().lower() or "auto"
    if want not in ("auto", "compiled", "pure"):
        raise ValueError(f"{ENV}={want!r}: expected 'compiled', 'pure' or unset")
    if want == "pure":
        return "pure", f"{ENV}=pure"
    mod, err = _probe()
    if mod is not None:
        return "compiled", f"d2shim._nashim loaded from {mod.EXTENSION_FILE} (nanoarrow {mod.NANOARROW_VERSION})"
    cause = f"{type(err).__name__}: {err}"
    if want == "compiled":
        raise ImportError(
            f"{ENV}=compiled but the compiled string backend is not available on "
            f"{platform.platform()} / CPython {platform.python_version()}: {cause}. "
            "Either install a wheel built for this platform, or unset "
            f"{ENV} to accept the pure-numba fallback."
        ) from err
    warnings.warn(
        f"d2shim: compiled string backend unavailable ({cause}); using the pure-numba "
        f"fallback. Answers are identical; single-record calls cost ~2x more. Set {ENV}=compiled "
        f"to make this an error, or {ENV}=pure to silence this warning.",
        FallbackWarning, stacklevel=3,
    )
    return "pure", f"fallback: {cause}"


def run_tree(float_cols, series_list, tree, patterns, out=None):
    """One entry point; the backend is chosen by which()."""
    name, _ = which()
    if name == "compiled":
        from .kernel_compiled import run_tree_compiled
        return run_tree_compiled(float_cols, series_list, tree, patterns, out=out)
    views = [arrowc.export(s) for s in series_list]
    try:
        return pure.run_tree(float_cols, views, tree, patterns, out=out)
    finally:
        for v in views:
            v.release()


def diagnose() -> dict:
    """What a `decider2 doctor` command would print: which backend, why, and
    the provenance of the binary. Safe to call on any platform."""
    name, reason = which()
    info = {
        "backend": name,
        "reason": reason,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "package_version": importlib.metadata.version("d2shim"),
    }
    mod, err = _probe()
    if mod is not None:
        info["extension"] = mod.EXTENSION_FILE
        info["nanoarrow"] = mod.NANOARROW_VERSION
    else:
        info["extension_error"] = f"{type(err).__name__}: {err}"
    try:
        from importlib.resources import files
        info["vendor_pin"] = files("d2shim").joinpath("vendor/VERSION").read_text().splitlines()[0]
    except Exception as e:  # pragma: no cover
        info["vendor_pin"] = f"unreadable: {e}"
    return info
