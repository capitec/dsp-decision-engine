from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import tempfile
from pathlib import Path

_HERE = Path(__file__).parent
_VENDOR = _HERE / "vendor"
_SOURCES = (_HERE / "c" / "shim.c", _VENDOR / "nanoarrow.c")
# The namespace prefixes every exported nanoarrow symbol, so another nanoarrow
# loaded into the process (pyarrow, adbc) can't interpose on ours.
_FLAGS = ("-O2", "-fPIC", "-shared", "-DNANOARROW_NAMESPACE=DeciderArrow")


def cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "decider" / "arrow-shim"


def build() -> Path:
    """The compiled shim, built on first use and cached by the content of its sources and flags."""
    # ponytail: POSIX `cc` only; Windows (MSVC) and a read-only cache dir need a prebuilt wheel.
    cc = os.environ.get("CC", "cc")
    key = hashlib.sha256(" ".join((cc, *_FLAGS, platform.machine())).encode())
    for path in (*_SOURCES, _VENDOR / "nanoarrow" / "nanoarrow.h"):
        key.update(path.read_bytes())
    out = cache_dir() / f"shim-{key.hexdigest()[:16]}.so"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".so", dir=out.parent)
    os.close(fd)
    try:
        proc = subprocess.run([cc, *_FLAGS, f"-I{_VENDOR}", *map(str, _SOURCES), "-o", tmp],
                              capture_output=True, text=True, check=False)
        if proc.returncode:
            raise RuntimeError(f"{cc} exited {proc.returncode}: {proc.stderr.strip()}")
        # Atomic, so concurrent first imports never load a half-written file.
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return out
