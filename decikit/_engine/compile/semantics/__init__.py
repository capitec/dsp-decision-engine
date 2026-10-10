import hashlib
from pathlib import Path

from decikit._engine.compile.semantics import decimals, numbers, strings  # noqa: F401  (imported for their registrations)

_DECIKIT = Path(__file__).resolve().parents[3]
_COMPILE = Path(__file__).resolve().parents[1]

# Numba keys a disk-cached step by its own bytecode only, so a change to an overload or a
# span type would otherwise keep serving stale machine code. Found by walking, so a new file is covered.
_FILES = sorted(
    {*Path(__file__).parent.rglob("*.py"), _COMPILE / "span.py", _COMPILE / "decimal_type.py",  # a rename must fail the import, not drop the file
     *(_DECIKIT / "ops").rglob("*.py")},
    key=lambda p: p.relative_to(_DECIKIT).as_posix(),
)
SALT = hashlib.sha256(b"".join(p.read_bytes() for p in _FILES)).hexdigest()
