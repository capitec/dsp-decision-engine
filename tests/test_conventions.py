import ast
import functools
import hashlib
import importlib
import re
from pathlib import Path

import pytest

from tests.conventions_allowlists import (
    BIG_ALL,
    KERNEL_DECORATOR,
    LAYER_CYCLE,
    NO_ALL,
    PRIVATE_IMPORT,
    SHARED_STATE,
    UPWARD_IMPORT,
)

DECIDER = Path(__file__).resolve().parent.parent / "decider"
MAX_LINES = 500
ALLOWLIST: dict[str, str] = {}  # {path relative to decider/: reason}
BANNED = re.compile(r"doc \d\d|§|EXPERIMENTS")


def _files():
    return sorted(DECIDER.rglob("*.py"))


def test_no_file_over_500_lines_unless_allowlisted():
    long = [
        f"{rel} ({n} lines)"
        for p in _files()
        if (rel := p.relative_to(DECIDER).as_posix()) not in ALLOWLIST
        and (n := len(p.read_text().splitlines())) > MAX_LINES
    ]
    assert not long, "split these into packages or allowlist them with a reason:\n" + "\n".join(long)


def test_no_references_to_docs_or_experiments():
    hits = [
        f"{p.relative_to(DECIDER)}:{i}: {line.strip()}"
        for p in _files()
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if BANNED.search(line)
    ]
    assert not hits, "code must not reference docs, sections or experiments:\n" + "\n".join(hits)


# --- public surface ---------------------------------------------------------

ENGINE = "_engine"
CORE = {"__init__", "steps", "ir", "trace", "debug", "contract", ENGINE}
ALL_BUDGET = 25
LAYERS = {
    0: "exceptions registry ops _annotations _settings _suggest _polars_docs fields types",
    1: "__init__ steps ir trace debug config _engine",
    2: "contract",
    3: "checks diff experiments data testing credit",
    4: "serving mcp cli _bridge _tools __main__",
}
LAYER = {name: n for n, names in LAYERS.items() for name in names.split()}
KERNEL_DECORATORS = {"overload", "overload_method", "overload_attribute", "overload_classmethod", "intrinsic"}
JITABLE = "register_jitable"
MUTATORS = {"update", "pop", "append", "add", "setdefault", "clear", "extend", "remove", "discard", "popitem",
            "insert", "sort", "reverse"}
LOCKED_METHODS = {"__init__", "_compile", "_build"}


def _is_private(segment):
    return segment.startswith("_") and not (segment.startswith("__") and segment.endswith("__"))


def _parts(p):
    return p.relative_to(DECIDER).with_suffix("").parts


def _dotted(p):
    return ".".join(_parts(p)).removesuffix(".__init__")


def _unit(p):
    return _parts(p)[0]


@functools.cache
def _tree(p):
    return ast.parse(p.read_text())


def _exists(module):
    base = DECIDER.parent.joinpath(*module.split("."))
    return base.with_suffix(".py").exists() or (base / "__init__.py").exists()


def _imports(p):
    """Yield (lineno, imported module, private name or None) for every decider import in a file, nested ones included."""
    package = ["decider", *_parts(p)[:-1]]
    for node in ast.walk(_tree(p)):
        if isinstance(node, ast.Import):
            mods = [(a.name, None) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = ".".join(package[: len(package) - node.level + 1]) if node.level else ""
            module = ".".join(x for x in (base, node.module) if x)
            mods = []
            for a in node.names:
                if _exists(f"{module}.{a.name}"):
                    mods.append((f"{module}.{a.name}", None))
                else:
                    mods.append((module, a.name if _is_private(a.name) else None))
        else:
            continue
        for module, name in mods:
            if module == "decider" or module.startswith("decider."):
                yield node.lineno, module, name


def _edges():
    """Every (importer file, line, imported module, private name or None, imported package or module unit)."""
    for p in _modules():
        for line, module, name in _imports(p):
            parts = module.split(".")
            yield p, line, module, name, parts[1] if len(parts) > 1 else "__init__"


def _fail(message, items):
    assert not items, message + ":\n" + "\n".join(sorted(items))


def _stale(allowlist, used, name):
    gone = sorted(set(allowlist) - set(used))
    assert not gone, f"{name} has entries that no longer apply, delete them:\n" + "\n".join(gone)


def _all_names(p):
    for node in _tree(p).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            return node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else []
    return None


def _modules():
    return [p for p in _files() if "templates" not in p.parts]


def _public(p):
    return p.name != "__main__.py" and not any(_is_private(s) for s in _parts(p))


def test_public_modules_define_all():
    missing = {p.relative_to(DECIDER).as_posix() for p in _modules() if _public(p) and _all_names(p) is None}
    _fail("add __all__ to these public modules, or make the module private with a leading underscore",
          missing - NO_ALL.keys())
    _stale(NO_ALL, missing, "NO_ALL")


def test_public_packages_stay_within_all_budget():
    big = {
        p.parent.relative_to(DECIDER).as_posix() or "."
        for p in _modules()
        if p.name == "__init__.py" and _public(p) and len(_all_names(p) or ()) > ALL_BUDGET
    }
    _fail(f"these packages export more than {ALL_BUDGET} names; trim __all__ or allowlist them", big - BIG_ALL.keys())
    _stale(BIG_ALL, big, "BIG_ALL")


def test_no_private_imports_across_packages():
    bad = {}
    for p, line, module, name, _ in _edges():
        parts = module.split(".")
        top = _unit(p)
        deep = any(_is_private(s) for s in parts[2:]) or name is not None
        engine_leak = len(parts) > 1 and parts[1] == ENGINE and top not in CORE
        if engine_leak or (deep and len(parts) > 1 and top != parts[1]):
            bad[f"{_dotted(p)} -> {module[len('decider.'):]}{'.' + name if name else ''}"] = line
    _fail("import the public name instead, or allowlist the import with a reason",
          [f"{k} (line {v})" for k, v in bad.items() if k not in PRIVATE_IMPORT])
    _stale(PRIVATE_IMPORT, bad, "PRIVATE_IMPORT")


# --- layers -----------------------------------------------------------------


def test_every_top_level_package_has_a_layer():
    units = {_unit(p) for p in _modules()}
    _fail("add these to LAYERS", units - LAYER.keys())


def test_imports_only_go_down_the_layer_map():
    bad = {
        f"{_dotted(p)} -> {module.removeprefix('decider.')} (line {line})": f"{_dotted(p)} -> {module.removeprefix('decider.')}"
        for p, line, module, _, unit in _edges()
        if LAYER[_unit(p)] < LAYER[unit]
    }
    _fail("move the import (or the code) down a layer, or allowlist it with a reason",
          [k for k, key in bad.items() if key not in UPWARD_IMPORT])
    _stale(UPWARD_IMPORT, bad.values(), "UPWARD_IMPORT")


def test_no_import_cycles_between_packages_in_a_layer():
    graph = {}
    for p, _, _, _, unit in _edges():
        src = _unit(p)
        if src != unit and LAYER[src] == LAYER[unit]:
            graph.setdefault(src, set()).add(unit)

    def reaches(a, b, seen=()):
        return a == b or any(reaches(n, b, (*seen, a)) for n in graph.get(a, ()) if n not in seen)

    cyclic = {f"{a} -> {b}" for a, targets in graph.items() for b in targets if reaches(b, a)}
    _fail("break these import cycles, or allowlist the edges you accept in LAYER_CYCLE", cyclic - LAYER_CYCLE.keys())
    _stale(LAYER_CYCLE, cyclic, "LAYER_CYCLE")


# --- shared objects ---------------------------------------------------------


def _runner_files():
    d = DECIDER / ENGINE / "run"
    return [p for p in [d / "engine.py", *sorted((d / "runners").glob("*.py"))] if p.exists()]


def _self_rooted(node):
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.value if not isinstance(node, ast.Call) else node.func
    return isinstance(node, ast.Name) and node.id == "self"


def _unlocked(node):
    for child in ast.iter_child_nodes(node):
        if not (isinstance(child, ast.With) and any(ast.unparse(i.context_expr) == "self._lock" for i in child.items)):
            yield child
            yield from _unlocked(child)


def _mutations(fn):
    for n in _unlocked(fn):
        if isinstance(n, (ast.Assign, ast.Delete)):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            targets = [n.target]
        else:
            targets = []
        for t in targets:
            if isinstance(t, (ast.Attribute, ast.Subscript)) and _self_rooted(t):
                yield n.lineno, ast.unparse(t)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in MUTATORS
                and isinstance(n.func.value, (ast.Attribute, ast.Subscript)) and _self_rooted(n.func.value)):
            yield n.lineno, ast.unparse(n.func)


def test_runners_and_executable_keep_no_per_call_state():
    bad = {}
    for p in _runner_files():
        for cls in (n for n in ast.walk(_tree(p)) if isinstance(n, ast.ClassDef)):
            for fn in (n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
                if fn.name not in LOCKED_METHODS:
                    for line, what in _mutations(fn):
                        bad[f"{p.relative_to(DECIDER).as_posix()}::{cls.name}.{fn.name}"] = f"{what} (line {line})"
    _fail("keep per-call state in locals or the State object; only __init__ and the locked compile may write self",
          [f"{k}: {v}" for k, v in bad.items() if k not in SHARED_STATE])
    _stale(SHARED_STATE, bad, "SHARED_STATE")


# --- kernel library ---------------------------------------------------------


def _kernel_dirs(p):
    rel = p.relative_to(DECIDER).as_posix()
    return rel.startswith(f"{ENGINE}/compile/semantics/"), rel.startswith("ops/")


def test_kernel_decorators_live_in_the_kernel_library():
    bad = {}
    for p in _modules():
        semantics, ops = _kernel_dirs(p)
        for n in ast.walk(_tree(p)):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for d in n.decorator_list:
                    d = d.func if isinstance(d, ast.Call) else d
                    name = d.attr if isinstance(d, ast.Attribute) else getattr(d, "id", "")
                    if (name in KERNEL_DECORATORS and not semantics) or (name == JITABLE and not (semantics or ops)):
                        bad.setdefault(p.relative_to(DECIDER).as_posix(), set()).add(name)
    _fail("register overloads and intrinsics in _engine/compile/semantics/ and jitable helpers in decider/ops/, or allowlist the file",
          [f"{k}: {sorted(v)}" for k, v in bad.items() if k not in KERNEL_DECORATOR])
    _stale(KERNEL_DECORATOR, bad, "KERNEL_DECORATOR")


def test_cache_salt_covers_every_semantics_and_ops_file():
    files = [p for d in (DECIDER / ENGINE / "compile" / "semantics", DECIDER / "ops") for p in sorted(d.rglob("*.py"))]
    if not files:
        pytest.skip("semantics/ and ops/ do not exist yet")
    salt = importlib.import_module(f"decider.{ENGINE}.compile.njit").SALT
    assert salt == hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest(), (
        "SALT must hash every file under semantics/ and ops/, found by walking the directories"
    )
