"""Importing a pipeline's files from their source as it is now."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


def load_module(file):
    """Import `file` from its source as it is now: by dotted name when it sits in a package, so its imports resolve."""
    # Cached bytecode is keyed on the file's mtime in whole seconds and its size, so an edit saved within the
    # same second as the last load, at the same length, would run the old code. Compile into a fresh cache.
    with tempfile.TemporaryDirectory(prefix="decider-pyc-") as fresh:
        before, sys.pycache_prefix = sys.pycache_prefix, fresh
        try:
            return _import(file)
        finally:
            sys.pycache_prefix = before


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
