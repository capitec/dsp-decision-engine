from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path

_VENDOR = Path(__file__).parent / "vendor"


def available() -> bool:
    """True when the compiled Arrow shim loads (building it first if needed) in this process."""
    try:
        from decider.engine.boundary._arrow import _shim  # noqa: F401
    except ImportError:
        return False
    return True


def diagnose() -> dict:
    """What is installed: interpreter, platform, the vendored nanoarrow pin, and the shim or its error.

    Never raises, so it works on a machine where the shim can't be built.

    >>> diagnose()["shim"]  # doctest: +SKIP
    'loaded'
    """
    info: dict = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        info["decider"] = importlib.metadata.version("decider")
    except importlib.metadata.PackageNotFoundError:
        info["decider"] = "not installed (running from source?)"
    info["vendor_pin"] = (_VENDOR / "VERSION").read_text().splitlines()[0]
    info["vendor_manifest"] = (
        "SHA512SUMS present" if (_VENDOR / "SHA512SUMS").is_file() else "SHA512SUMS missing"
    )
    try:
        from decider.engine.boundary._arrow import _shim
    except ImportError as exc:
        info["shim"] = "unavailable"
        info["error"] = str(exc)
    else:
        info["shim"] = "loaded"
        info.update(_shim.info())
    return info
