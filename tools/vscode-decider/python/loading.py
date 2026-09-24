"""Importing a pipeline's files from their source as it is now, and diffing a step against the code that ran."""
from __future__ import annotations

import difflib
import importlib.util
import inspect
import sys
import tempfile
from pathlib import Path


# Each loaded module file's text as it was loaded, so an edit can be diffed against the code that ran.
TEXTS: dict[str, str] = {}


def load_module(file):
    """Import `file` from its source as it is now: by dotted name when it sits in a package, so its imports resolve."""
    # Cached bytecode is keyed on the file's mtime in whole seconds and its size, so an edit saved within the
    # same second as the last load, at the same length, would run the old code. Compile into a fresh cache.
    with tempfile.TemporaryDirectory(prefix="decider-pyc-") as fresh:
        before, sys.pycache_prefix = sys.pycache_prefix, fresh
        try:
            mod = _import(file)
        finally:
            sys.pycache_prefix = before
    top = mod.__name__.split(".")[0]
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and (m.__name__ == top or m.__name__.startswith(top + ".")):
            TEXTS[f] = Path(f).read_text()
    return mod


def _import(file):
    path = Path(file).resolve()
    root, parts = path.parent, [path.stem]
    while (root / "__init__.py").exists():
        parts.insert(0, root.name)
        root = root.parent
    sys.path.insert(0, str(root))
    name = ".".join(parts)
    if len(parts) > 1:
        for mod_name in [m for m in sys.modules if m == parts[0] or m.startswith(parts[0] + ".")]:
            del sys.modules[mod_name]  # a fresh import, so edits since the last describe count
        return importlib.import_module(name)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # classes defined in it (ConfigurableSteps) resolve by import path
    spec.loader.exec_module(mod)
    return mod


def source_diff(old, old_texts, new, new_texts):
    """The lines that differ between two steps' Python, each read from its file as it was loaded, as `-`/`+` lines."""
    def lines(step_, texts):
        code = getattr(getattr(step_, "fn", None), "__code__", None)
        text = code and texts.get(code.co_filename)
        if not text:
            return []
        return [line.rstrip("\n") for line in inspect.getblock(text.splitlines(True)[code.co_firstlineno - 1:])]
    return [line for line in difflib.unified_diff(lines(old, old_texts), lines(new, new_texts), lineterm="", n=0)
            if line[:1] in "+-" and not line.startswith(("+++", "---"))]
