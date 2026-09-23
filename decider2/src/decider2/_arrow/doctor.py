"""`diagnose()`: what a future `decider2 doctor` prints. Importable and
callable on a box where the extension is absent or broken — it reports
that, rather than raising."""
from __future__ import annotations

import importlib.metadata
import platform
import sys
from importlib.resources import files


def available() -> bool:
    """True when the compiled shim loads in this process."""
    try:
        from decider2._arrow import _shim  # noqa: F401
    except ImportError:
        return False
    return True


def diagnose() -> dict:
    """Interpreter, platform, package version, the vendored nanoarrow pin,
    and either the loaded extension's details or the ImportError it
    produced."""
    info: dict = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        info["decider2"] = importlib.metadata.version("decider2")
    except importlib.metadata.PackageNotFoundError:
        info["decider2"] = "not installed (running from source?)"
    vendor = files("decider2._arrow").joinpath("vendor")
    try:
        info["vendor_pin"] = vendor.joinpath("VERSION").read_text().splitlines()[0]
        info["vendor_manifest"] = "SHA512SUMS present" if vendor.joinpath("SHA512SUMS").is_file() \
            else "SHA512SUMS missing"
    except Exception as exc:  # pragma: no cover
        info["vendor_pin"] = f"unreadable: {exc}"
    try:
        from decider2._arrow import _shim
    except ImportError as exc:
        info["shim"] = "unavailable"
        info["error"] = str(exc)
    else:
        info["shim"] = "loaded"
        info.update(_shim.info())
    return info
