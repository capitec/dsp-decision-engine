"""Real-file, content-addressed driver cache — doc 05 §4.1, §4.2.

Two rules this module exists to enforce, because violating either one is a
*silent wrong-answer* bug rather than a loud one (EXPERIMENTS.md §C, §K, §M):

1. **Never `exec`.** Numba cannot cache a function with no source file —
   `RuntimeError: cannot cache function: no locator available for file
   '<string>'`, and it fires at decoration, not at first call. So every
   generated driver is written to a real `.py` file before anything imports
   it (doc 05 §4.1).
2. **Content-addressed names, imported by module name — never
   `importlib.util.spec_from_file_location`, never a normalised mtime.**
   EXPERIMENTS.md §J2: the numba cache does **not** survive a fresh process
   when the generated driver is loaded by file location, even with
   byte-identical bytes and a preserved mtime; imported by module name (the
   directory on `sys.path`, `importlib.import_module`), it does — 12.1 s
   cold, ~0.2 s on every subsequent process. And EXPERIMENTS.md §C: normalise
   mtimes for "reproducibility" and a same-slot constant edit becomes a
   cache **hit** that serves the pre-edit answer, because the index key
   hashes `co_code`, which excludes `co_consts`. Content addressing sidesteps
   both failure modes structurally: same bytes -> same name -> same cache
   entry; different bytes -> different name -> a miss, never a stale hit.

Doc 05 §4.2 lists seven conditions a build-time cache entry must meet to be
reused at runtime. Six are numba's own bookkeeping (directory, basename,
`def`-line, exact `(st_mtime, st_size)`, `magic_tuple`, `co_code`/closure
hash) and this module doesn't touch them. The two that *are* this module's
job:

- condition 2 (same basename) — guaranteed by construction: identical source
  hashes to the identical filename, always.
- condition 7 (same `sys.modules` registration name) — derived from the
  *same* hash as the filename, never a counter or anything random, so it is
  as stable across processes as the filename is.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
import threading
import types as pytypes
from dataclasses import dataclass
from pathlib import Path

_LOCK = threading.Lock()


def content_hash(source: str) -> str:
    """The address. Source text in, a stable hex digest out.

    Never a timestamp, never `id()`, never a PID (doc 05 §4.2) — sha256 of
    the UTF-8 bytes is the entire input, so the same source always produces
    the same digest, in this process or the next one.
    """
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CachedModule:
    """A generated driver, written to disk and resident in `sys.modules`."""

    digest: str
    path: Path
    module_name: str
    module: object


def _module_name_for(digest: str, namespace: str) -> str:
    # Condition 7 (doc 05 §4.2): derived from the same hash as the filename.
    return f"{namespace}._gen_{digest}"


def _ensure_namespace_package(pkg_name: str, build_dir: Path) -> None:
    """Register `build_dir` as the package whose submodules are the
    generated drivers, so `importlib.import_module("<pkg>._gen_<hash>")`
    resolves without ever calling `spec_from_file_location` — the import
    path EXPERIMENTS.md §J2 measured as the one that survives a fresh
    process."""
    build_dir_str = str(build_dir)
    existing = sys.modules.get(pkg_name)
    if existing is None:
        pkg = pytypes.ModuleType(pkg_name)
        pkg.__path__ = [build_dir_str]
        sys.modules[pkg_name] = pkg
        return
    if build_dir_str not in list(existing.__path__):
        existing.__path__ = list(existing.__path__) + [build_dir_str]


def get_or_build(
    source: str,
    build_dir: "str | Path",
    *,
    namespace: str = "decider2_generated",
) -> CachedModule:
    """Write `source` to a content-addressed file, if it isn't already
    there, and import it by module name.

    Idempotent and safe across processes: two calls with byte-identical
    `source` land on the same path and the same `sys.modules` name, whether
    or not *this* process is the one that wrote the file — which is what
    lets a build-time cache survive a runtime restart (doc 00-BUILD.md §2,
    "Lifecycle"). Two calls with different `source` never collide, because
    the name changes with the content: a content-only edit that keeps the
    file size and mtime identical still produces a *new* file here, so it
    misses by construction rather than risking the stale-`co_consts` hit
    EXPERIMENTS.md §C measured.
    """
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    digest = content_hash(source)
    filename = f"_gen_{digest}.py"
    path = build_dir / filename
    module_name = _module_name_for(digest, namespace)

    with _LOCK:
        if not path.exists():
            tmp = path.with_suffix(".py.tmp")
            tmp.write_text(source, encoding="utf-8")
            tmp.replace(path)  # atomic rename on the same filesystem

        cached = sys.modules.get(module_name)
        if cached is not None:
            return CachedModule(digest, path, module_name, cached)

        _ensure_namespace_package(namespace, build_dir)
        module = importlib.import_module(module_name)
        return CachedModule(digest, path, module_name, module)


def list_cached(build_dir: "str | Path") -> list[str]:
    """Every content-addressed driver file already on disk under
    `build_dir` — a `decider2 build --verify`-style check reads this to
    assert a runtime load triggers zero new files, i.e. zero compiles
    (doc 02 §3.4)."""
    build_dir = Path(build_dir)
    if not build_dir.exists():
        return []
    return sorted(p.name for p in build_dir.glob("_gen_*.py"))
