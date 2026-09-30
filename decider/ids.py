"""Durable step and flow ids: a source-rewriting generator that inserts `id=` tokens.

The generator is a development-time tool. It adds an opaque `id="0123abcdef45"`
keyword to every scalar step and every named flow declaration that lacks one,
changing no other byte. It refuses to run unless the working tree is clean,
parses the whole tree before writing anything, and reports a malformed or
duplicate id by name and location instead of editing source.

Pure helpers (`add_ids`, `_find_decls`) are separated from the git-orchestrated
`generate` so they can be tested without a repository.
"""
from __future__ import annotations

import ast
import os
import secrets
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from decider.engine.ir.origin import check_id, check_name
from decider.exceptions import WiringError

_SCALAR = {"step", "frame_step"}
_COMPOSITE = {"flow", "dag", "branch", "loop", "each", "optimise"}
_CONSTRUCTORS = _SCALAR | _COMPOSITE
_DECORATORS = {"step", "frame_step"}
_EXCLUDED = {".git", "__pycache__", ".venv", "venv", "node_modules",
             ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages", "dist", "build"}


class IdError(Exception):
    """The generator refused to run; `remediation` says what to do next."""

    def __init__(self, message: str, remediation: str):
        super().__init__(message)
        self.remediation = remediation


@dataclass(frozen=True)
class Report:
    """What a run did: each change as `(path, lineno, name, new_id)` and the file count."""

    changes: list[tuple[Path, int, str | None, str]]
    files: int


def gen_id() -> str:
    return secrets.token_hex(6)


def _offsets(source: str) -> list[int]:
    offs = [0]
    for line in source.splitlines(keepends=True):
        offs.append(offs[-1] + len(line))
    return offs


def _line_indent(source: str, offs: list[int], lineno: int) -> str:
    start = offs[lineno - 1]
    end = start
    while end < len(source) and source[end] in " \t":
        end += 1
    return source[start:end]


def _kwarg(call: ast.Call, arg: str) -> ast.keyword | None:
    return next((kw for kw in call.keywords if kw.arg == arg), None)


def _kwarg_str(call: ast.Call, arg: str) -> str | None:
    kw = _kwarg(call, arg)
    if kw is not None and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
        return kw.value.value
    return None


def _dec_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node, ast.Name):
        return node.id
    return None


@dataclass(frozen=True)
class _Decl:
    node: ast.AST            # ast.Call, or ast.Name for a bare @step
    kind: str                # "scalar" or "composite"
    bare: bool               # bare @step / @frame_step decorator
    name: str | None         # literal `name=`, else the function's name
    has_id: bool             # an `id=` keyword is present
    id: str | None           # its string value when it is a plain string literal
    id_node: ast.Constant | None
    lineno: int


def _decl_from_call(call: ast.Call, kind: str, fn_name: str | None) -> _Decl:
    name = _kwarg_str(call, "name")
    if name is None and kind == "scalar":
        name = fn_name
    id_kw = _kwarg(call, "id")
    if id_kw is not None and isinstance(id_kw.value, ast.Constant) and isinstance(id_kw.value.value, str):
        id_value, id_node = id_kw.value.value, id_kw.value
    else:
        id_value, id_node = None, None
    return _Decl(call, kind, False, name, id_kw is not None, id_value, id_node, call.lineno)


def _find_decls(tree: ast.AST) -> list[_Decl]:
    decls: list[_Decl] = []
    decorator_calls: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if _dec_name(dec) in _DECORATORS:
                    if isinstance(dec, ast.Name):
                        decls.append(_Decl(dec, "scalar", True, node.name, False, None, None, dec.lineno))
                    else:
                        decorator_calls.add(id(dec))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if id(dec) in decorator_calls:
                    decls.append(_decl_from_call(dec, "scalar", node.name))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in _CONSTRUCTORS and id(node) not in decorator_calls):
            kind = "scalar" if node.func.id in _SCALAR else "composite"
            if kind == "composite" and _kwarg(node, "name") is None:
                continue  # an anonymous flow stays id-less and transparent
            decls.append(_decl_from_call(node, kind, None))
    return sorted(decls, key=lambda d: (d.node.lineno, d.node.col_offset))


def _last_arg(call: ast.Call) -> ast.expr:
    nodes = list(call.args) + [kw.value for kw in call.keywords]
    return max(nodes, key=lambda n: (n.lineno, n.col_offset))


def _last_arg_end(offs: list[int], call: ast.Call) -> int:
    last = _last_arg(call)
    return offs[last.end_lineno - 1] + last.end_col_offset


def _insert_kwarg(source: str, offs: list[int], call: ast.Call, fragment: str) -> tuple[int, int, str]:
    close = offs[call.end_lineno - 1] + call.end_col_offset - 1
    if call.lineno == call.end_lineno:
        sep = "" if not (call.args or call.keywords) else ", "
        return close, close, sep + fragment
    if not (call.args or call.keywords):
        pos = offs[call.lineno - 1] + call.func.end_col_offset + 1  # right after the opening paren
        return pos, pos, fragment + ","
    arg_end = _last_arg_end(offs, call)
    arg_indent = _line_indent(source, offs, _last_arg(call).lineno)
    if "\n" in source[arg_end:close]:
        paren_indent = _line_indent(source, offs, call.end_lineno)
    else:
        paren_indent = _line_indent(source, offs, call.lineno)
    return arg_end, close, ",\n" + arg_indent + fragment + ",\n" + paren_indent


def _apply(source: str, edits: Iterable[tuple[int, int, str]]) -> str:
    # (start, end, text); end == start is an insert at start. Right-to-left so offsets stay valid.
    for start, end, text in sorted(edits, key=lambda e: e[0], reverse=True):
        source = source[:start] + text + source[end:]
    return source


def add_ids(source: str, id_for: Callable[[_Decl], str] | None = None) -> tuple[str, list[tuple[int, str | None, str]]]:
    """Return `source` with `id="…"` added to each declaration missing one, and the changes as `(lineno, name, id)`.

    Example::

        text, changes = add_ids(open("pipeline.py").read())
    """
    id_for = id_for or (lambda _d: gen_id())
    offs = _offsets(source)
    edits: list[tuple[int, int, str]] = []
    changes: list[tuple[int, str | None, str]] = []
    for decl in _find_decls(ast.parse(source)):
        if decl.has_id:
            continue
        id_ = id_for(decl)
        if decl.bare:
            pos = offs[decl.node.lineno - 1] + decl.node.end_col_offset
            edits.append((pos, pos, f'(id="{id_}")'))
        else:
            edits.append(_insert_kwarg(source, offs, decl.node, f'id="{id_}"'))
        changes.append((decl.lineno, decl.name, id_))
    return _apply(source, edits), changes


def _read(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def _py_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".py" else []
    return [p for p in sorted(path.rglob("*.py"))
            if not any(part.startswith(".") or part in _EXCLUDED for part in p.parts)]


def _git(dir_: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(dir_), *args], capture_output=True, text=True)


def _git_root(dir_: Path) -> Path:
    r = _git(dir_, "rev-parse", "--show-toplevel")
    if r.returncode != 0:
        raise IdError(f"{dir_} is not inside a git repository",
                      "run `git init` and commit the pipeline first")
    return Path(r.stdout.strip())


def _require_clean(root: Path) -> None:
    r = _git(root, "status", "--porcelain")
    if r.returncode != 0:
        raise IdError("git status failed", "check that git works in this directory")
    dirty = [line for line in r.stdout.splitlines() if line.strip()]
    if dirty:
        raise IdError(
            "working tree is not clean; refusing to rewrite source:\n  " + "\n  ".join(dirty),
            "commit or stash these changes first (untracked files: `git add` then commit), then re-run",
        )


def _validate(decls: dict[Path, list[_Decl]]) -> None:
    for path, ds in decls.items():
        for d in ds:
            loc = f"{path}:{d.lineno}"
            label = d.name or type(d.node).__name__
            if d.name is not None:
                try:
                    check_name(d.name)
                except WiringError as e:
                    raise IdError(f"{loc}: {e}",
                                  "rename the step (names must be non-empty and contain no '/' or '#')") from e
            if d.has_id and d.id is None:
                raise IdError(f"{loc}: step {label!r} has an id that is not a plain string literal",
                              f'set id to a 12-hex string literal like id="0123abcdef45"')
            if d.id is not None:
                try:
                    check_id(d.id)
                except WiringError as e:
                    raise IdError(f"{loc}: step {label!r}: {e}",
                                  "fix or remove the malformed id; ids are 12 lowercase hex characters")


def _duplicates(decls: dict[Path, list[_Decl]]) -> dict[str, list[tuple[Path, _Decl]]]:
    by_id: dict[str, list[tuple[Path, _Decl]]] = {}
    for path, ds in decls.items():
        for d in ds:
            if d.id is not None:
                by_id.setdefault(d.id, []).append((path, d))
    return {i: occ for i, occ in by_id.items() if len(occ) > 1}


def _fresh_id(used: set[str]) -> str:
    while True:
        id_ = gen_id()
        if id_ not in used:
            used.add(id_)
            return id_


def _atomic_write(path: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def generate(path: str | Path = ".", *, fix: bool = False, check: bool = False) -> Report:
    """Add `id=` to every step and named flow under `path`, and write the result.

    Refuses to run unless the tree is clean, parses the whole tree first, and
    aborts (before writing anything) on a syntax error, malformed id, invalid
    step name or duplicate id. With `fix`, a duplicate id is re-assigned to its
    later copies; with `check`, nothing is written.

    Example::

        generate("credit")   # adds ids to every step/flow in credit/
    """
    dir_ = Path(path)
    dir_ = dir_ if dir_.is_dir() else dir_.parent
    root = _git_root(dir_)
    _require_clean(root)
    files = _py_files(Path(path))
    contents = {f: _read(f) for f in files}
    trees: dict[Path, ast.AST] = {}
    for f, src in contents.items():
        try:
            trees[f] = ast.parse(src)
        except SyntaxError as e:
            raise IdError(f"{f}:{e.lineno}:{e.offset}: syntax error: {e.msg}", "fix the syntax error and re-run") from e
    decls = {f: _find_decls(t) for f, t in trees.items()}
    _validate(decls)
    dups = _duplicates(decls)
    if dups and not fix:
        lines = [f"duplicate id {id_!r} on {len(occ)} steps:" for id_, occ in dups.items()]
        for id_, occ in dups.items():
            for f, d in occ:
                lines.append(f"  {f}:{d.lineno}  {d.name or '(anonymous)'}")
        raise IdError("\n".join(lines),
                      "these look copy/pasted; run with --fix to assign a fresh id to the later copies, "
                      "or edit one by hand")
    used = {d.id for ds in decls.values() for d in ds if d.id is not None}
    assigned: dict[int, str] = {}
    seen = set()
    for f in files:
        for d in decls[f]:
            if d.id is None:
                assigned[id(d)] = _fresh_id(used)
            elif d.id in dups and fix and d.id in seen:
                assigned[id(d)] = _fresh_id(used)
            else:
                seen.add(d.id)
    changes: list[tuple[Path, int, str | None, str]] = []
    new_contents: dict[Path, str] = {}
    for f in files:
        src = contents[f]
        offs = _offsets(src)
        edits: list[tuple[int, int, str]] = []
        for d in decls[f]:
            if id(d) not in assigned:
                continue
            id_ = assigned[id(d)]
            changes.append((f, d.lineno, d.name, id_))
            if d.bare:
                pos = offs[d.node.lineno - 1] + d.node.end_col_offset
                edits.append((pos, pos, f'(id="{id_}")'))
            elif d.id_node is not None:
                start = offs[d.id_node.lineno - 1] + d.id_node.col_offset
                end = offs[d.id_node.end_lineno - 1] + d.id_node.end_col_offset
                edits.append((start, end, f'"{id_}"'))
            else:
                edits.append(_insert_kwarg(src, offs, d.node, f'id="{id_}"'))
        if edits:
            new_contents[f] = _apply(src, edits)
    if not check:
        for f, new in new_contents.items():
            _atomic_write(f, new)
    return Report(changes=changes, files=len(new_contents))
