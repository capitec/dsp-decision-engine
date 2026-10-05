"""Importing a pipeline's files from their source as it is now."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

from decider.config import JsonFileStore
from decider.data.providers import project_root  # noqa: F401  (re-exported for bridge.py)


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


def is_flow_file(file):
    return str(file).endswith((".yaml", ".yml", ".json"))


def flow_name(file):
    from decider.flowfile.load import _parse

    return _parse(Path(file))["flow"]


def flow_components(file):
    """The module a flow file's `components:` names, imported from beside the file; a blank module when it names none."""
    from decider.flowfile.load import _parse

    path = Path(file).resolve()
    project = path.parent.parent if path.parent.name == "flows" else path.parent
    root = project
    while (root / "__init__.py").exists():
        root = root.parent
    for p in dict.fromkeys([str(root), str(project)]):
        if p not in sys.path:
            sys.path.insert(0, p)
    name = _parse(path).get("components")
    return importlib.import_module(name) if name else types.ModuleType(path.stem)


def latest_config(root):
    if root is None:
        return {}
    store = JsonFileStore(basepath=str(root / "configs"))
    version = store.latest_version()
    return store.read(version).config if version else {}


def components(file):
    """The names a flow file's `step:` nodes can use: its components' functions and steps, then its config's documents."""
    from decider.flowfile.load import _components_index

    docs = [n for n, d in latest_config(project_root(file)).items() if n != "params" and isinstance(d, dict) and "type" in d]
    return sorted(_components_index(flow_components(file))) + sorted(docs)
