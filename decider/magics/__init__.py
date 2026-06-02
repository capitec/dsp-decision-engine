"""
Jupyter magic for interactive module development.

Load in a notebook with:
    %load_ext decider.magics

Then define a module cell:
    %%module CreditScorer

    class CreditScorerConfig(BaseModel):
        weight: float = 1.0

    def score(amount: pl.Expr, config: CreditScorerConfig) -> pl.Expr:
        return amount * config.weight

This writes (or overwrites) the extension file, reloads it, and injects
the generated class into the notebook namespace.
"""

import importlib
import importlib.util
import os
import re
import sys
import textwrap
import typing as t
from pathlib import Path


_EXTENSION_TEMPLATE = '''\
import polars as pl
from pydantic import BaseModel
from decider.modules.functional import generate_from_functions
from decider.modules import register_graph_module

{user_code}

{class_name} = generate_from_functions(
    {type_id!r},
    {function_names},
)

register_graph_module({class_name})
'''


def _to_snake(name: str) -> str:
    """CreditScorer → credit_scorer"""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()


def _extract_top_level_function_names(source: str) -> t.List[str]:
    """Return names of top-level `def` statements in source, in order."""
    names = []
    for m in re.finditer(r"^def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", source, re.MULTILINE):
        names.append(m.group(1))
    return names


def _write_extension(ext_dir: Path, class_name: str, user_code: str) -> Path:
    snake = _to_snake(class_name)
    type_id = snake

    function_names = _extract_top_level_function_names(user_code)
    if not function_names:
        raise ValueError(
            f"%%module {class_name}: no top-level functions found in cell body. "
            "Define at least one `def` that returns pl.Expr."
        )

    fn_list = ",\n    ".join(function_names)
    source = _EXTENSION_TEMPLATE.format(
        user_code=textwrap.dedent(user_code).strip(),
        class_name=class_name,
        type_id=type_id,
        function_names=fn_list,
    )

    pkg_dir = ext_dir / snake
    pkg_dir.mkdir(parents=True, exist_ok=True)
    init_file = pkg_dir / "__init__.py"
    init_file.write_text(source)
    return init_file


def _reload_extension(ext_dir: Path, class_name: str) -> t.Any:
    """Import (or reload) the extension package and return the class."""
    snake = _to_snake(class_name)
    ext_dir_str = str(ext_dir)

    if ext_dir_str not in sys.path:
        sys.path.insert(0, ext_dir_str)

    if snake in sys.modules:
        mod = importlib.reload(sys.modules[snake])
    else:
        mod = importlib.import_module(snake)

    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ImportError(
            f"%%module {class_name}: after writing the extension, "
            f"could not find '{class_name}' in {snake}/__init__.py"
        )
    return cls


def _find_ext_dir(ip) -> Path:
    """
    Resolve the extensions directory.  Priority:
    1. ip.user_ns['DECIDER_EXTENSIONS_DIR']  (user sets this once in the notebook)
    2. A 'decider_extensions/' folder next to the notebook file
    3. 'decider_extensions/' in the current working directory
    """
    if "DECIDER_EXTENSIONS_DIR" in ip.user_ns:
        return Path(ip.user_ns["DECIDER_EXTENSIONS_DIR"]).resolve()

    # Try to locate the notebook's directory
    nb_file = getattr(ip, "starting_dir", None) or os.getcwd()
    candidate = Path(nb_file) / "decider_extensions"
    if candidate.exists():
        return candidate.resolve()

    return (Path(os.getcwd()) / "decider_extensions").resolve()


def _register_with_graph_module(cls) -> None:
    from decider.modules import register_graph_module
    register_graph_module(cls)


def module_magic(line: str, cell: str = None):
    """
    %%module ClassName

    Writes/updates decider_extensions/<snake_class>/__init__.py, reloads it,
    and injects ClassName into the notebook namespace.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except ImportError:
        ip = None

    class_name = line.strip()
    if not class_name:
        raise ValueError("Usage: %%module ClassName")

    if cell is None:
        # Called as line magic — print help
        print(f"Usage:\n  %%%%module {class_name}\n  <function definitions>")
        return

    ext_dir = _find_ext_dir(ip) if ip else Path(os.getcwd()) / "decider_extensions"

    init_file = _write_extension(ext_dir, class_name, cell)
    print(f"[module] Written: {init_file}")

    cls = _reload_extension(ext_dir, class_name)
    _register_with_graph_module(cls)
    print(f"[module] {class_name} registered  type={cls._CLASS_TYPE_IDENTIFIER!r}")

    if ip is not None:
        ip.user_ns[class_name] = cls
        print(f"[module] {class_name} available in notebook namespace")


def load_ipython_extension(ip):
    ip.register_magic_function(module_magic, magic_kind="line_cell", magic_name="module")
