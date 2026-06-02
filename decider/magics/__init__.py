"""
Jupyter magic for interactive module development.

Load in a notebook with:
    %load_ext decider.magics

Inline module (writes decider_extensions/<snake>/__ init__.py):
    %%module CreditScorer

    def score(amount: pl.Expr) -> pl.Expr:
        return amount * 2

Shared package module (writes decider_extensions/mylib/src/mylib/credit_scorer.py):
    %%module CreditScorer --package mylib

    def score(amount: pl.Expr) -> pl.Expr:
        return amount * 2

Override extension directory for the session:
    DECIDER_EXTENSIONS_DIR = "/path/to/decider_extensions"
"""

import argparse
import importlib
import os
import sys
import typing as t
from pathlib import Path

from decider.templates.scaffold import (
    to_snake,
    write_inline_module,
    write_package_module,
)


# ── extension directory resolution ───────────────────────────────────────────

def _find_ext_dir(ip) -> Path:
    """
    Priority:
    1. ip.user_ns['DECIDER_EXTENSIONS_DIR']
    2. decider_extensions/ next to the notebook
    3. decider_extensions/ in cwd
    """
    if ip is not None and "DECIDER_EXTENSIONS_DIR" in ip.user_ns:
        return Path(ip.user_ns["DECIDER_EXTENSIONS_DIR"]).resolve()
    nb_dir = Path(getattr(ip, "starting_dir", None) or os.getcwd()) if ip else Path(os.getcwd())
    candidate = nb_dir / "decider_extensions"
    if candidate.exists():
        return candidate.resolve()
    return (Path(os.getcwd()) / "decider_extensions").resolve()


# ── import helpers ────────────────────────────────────────────────────────────

def _ensure_on_path(directory: Path) -> None:
    s = str(directory)
    if s not in sys.path:
        sys.path.insert(0, s)


def _import_or_reload(module_name: str) -> t.Any:
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _load_class_from_inline(ext_dir: Path, class_name: str) -> type:
    _ensure_on_path(ext_dir)
    mod = _import_or_reload(to_snake(class_name))
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ImportError(f"Could not find {class_name!r} in {mod.__file__}")
    return cls


def _load_class_from_package(ext_dir: Path, class_name: str, package_name: str) -> type:
    # The importable root for a src-layout package is ext_dir/<pkg>/src/
    src_root = ext_dir / package_name / "src"
    _ensure_on_path(src_root)
    snake = to_snake(class_name)
    # import the submodule directly so we get the freshest version
    full_name = f"{package_name}.{snake}"
    if full_name in sys.modules:
        mod = importlib.reload(sys.modules[full_name])
    else:
        mod = importlib.import_module(full_name)
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ImportError(f"Could not find {class_name!r} in {mod.__file__}")
    return cls


def _register(cls: type) -> None:
    from decider.modules import register_graph_module
    register_graph_module(cls)


# ── argument parsing ──────────────────────────────────────────────────────────

def _parse_line(line: str) -> tuple[str, t.Optional[str]]:
    """Return (class_name, package_name_or_None)."""
    parts = line.split()
    if not parts:
        raise ValueError("Usage: %%module ClassName [--package pkg_name]")
    class_name = parts[0]
    package_name = None
    rest = parts[1:]
    i = 0
    while i < len(rest):
        if rest[i] in ("--package", "-p") and i + 1 < len(rest):
            package_name = rest[i + 1]
            i += 2
        else:
            i += 1
    return class_name, package_name


# ── magic implementation ──────────────────────────────────────────────────────

def module_magic(line: str, cell: str = None):
    """
    %%module ClassName [--package pkg]

    Writes/updates the extension file, reloads it, registers the class with
    GraphModule, and injects it into the notebook namespace.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except ImportError:
        ip = None

    class_name, package_name = _parse_line(line)

    if cell is None:
        print(f"Usage:\n  %%%%module {class_name} [--package pkg_name]\n  <function definitions>")
        return

    ext_dir = _find_ext_dir(ip)

    if package_name:
        module_file, init_file = write_package_module(ext_dir, class_name, package_name, cell)
        print(f"[module] Written:  {module_file}")
        print(f"[module] Updated:  {init_file}")
        cls = _load_class_from_package(ext_dir, class_name, package_name)
    else:
        init_file = write_inline_module(ext_dir, class_name, cell)
        print(f"[module] Written:  {init_file}")
        cls = _load_class_from_inline(ext_dir, class_name)

    _register(cls)
    print(f"[module] {class_name} registered  type={cls._CLASS_TYPE_IDENTIFIER!r}")

    if ip is not None:
        ip.user_ns[class_name] = cls
        print(f"[module] {class_name} injected into namespace")


def load_ipython_extension(ip):
    ip.register_magic_function(module_magic, magic_kind="line_cell", magic_name="module")
