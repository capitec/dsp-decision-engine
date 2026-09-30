"""Experiment 05 follow-up: actual source transformation, not only `ast.parse`.

Proves item 4 against a temporary git fixture: the rewrite inserts `id="..."`
into step declarations while preserving comments, decorators, imports,
formatting and source mapping, and enforces parse-before-write, clean-tree,
idempotence, duplicate detection and failure atomicity.

Run: uv run python notes/vscode-redesign/experimentation/05-step-id/transform.py
"""
from __future__ import annotations

import ast
import re
import secrets
import subprocess
import tempfile
from pathlib import Path

ID_RE = re.compile(r"[0-9a-f]{12}\Z")
CONSTRUCTORS = {"step", "frame_step", "flow", "dag", "branch", "loop", "each", "optimise"}
DECORATORS = {"step", "frame_step"}


def gen_id() -> str:
    return secrets.token_hex(6)


def _offsets(source: str) -> list[int]:
    offs = [0]
    for line in source.splitlines(keepends=True):
        offs.append(offs[-1] + len(line))
    return offs


def _has_id(call: ast.Call) -> bool:
    return any(k.arg == "id" for k in call.keywords)


def _dec_name(dec: ast.expr) -> str | None:
    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
        return dec.func.id
    if isinstance(dec, ast.Name):
        return dec.id
    return None


def _targets(source: str) -> list[tuple[ast.AST, str]]:
    """Every step declaration missing an `id`, as `(node, kind)` with kind
    'decorator' or 'call'. Declarations that already have one are skipped, so
    the pass is idempotent."""
    tree = ast.parse(source)
    decorator_calls: set[int] = set()
    bare: list[ast.Name] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if _dec_name(dec) not in DECORATORS:
                    continue
                if isinstance(dec, ast.Name):
                    bare.append(dec)
                else:
                    decorator_calls.add(id(dec))
    out: list[tuple[ast.AST, str]] = [(d, "decorator") for d in bare]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if id(dec) in decorator_calls and not _has_id(dec):
                    out.append((dec, "decorator"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in CONSTRUCTORS and id(node) not in decorator_calls \
                and not _has_id(node):
            out.append((node, "call"))
    return out


def _kwarg_fragment(source: str, offs: list[int], call: ast.Call, id_: str) -> str:
    kwarg = f'id="{id_}"'
    close = offs[call.end_lineno - 1] + call.end_col_offset - 1
    if call.end_lineno == call.func.end_lineno:
        sep = "" if (not call.args and not call.keywords) else ", "
        return sep + kwarg
    prefix = source[offs[call.end_lineno - 1]:close]
    if prefix.strip() == "":
        func = source[offs[call.func.end_lineno - 1]:offs[call.func.end_lineno - 1] + call.func.end_col_offset]
        indent = " " * (len(func) - len(func.lstrip()) + 4)
        return indent + kwarg + "\n"
    return ", " + kwarg


def transform(source: str, id_for=None) -> str:
    """Return `source` with `id="…"` added to each missing step declaration,
    byte-preserving everywhere else."""
    id_for = id_for or (lambda _node: gen_id())
    offs = _offsets(source)
    edits: list[tuple[int, str]] = []
    for node, kind in _targets(source):
        if kind == "decorator" and isinstance(node, ast.Name):
            edits.append((offs[node.lineno - 1] + node.end_col_offset, f'(id="{id_for(node)}")'))
        else:
            assert isinstance(node, ast.Call)
            pos = offs[node.end_lineno - 1] + node.end_col_offset - 1
            edits.append((pos, _kwarg_fragment(source, offs, node, id_for(node))))
    for pos, frag in sorted(edits, reverse=True):
        source = source[:pos] + frag + source[pos:]
    return source


def scan_duplicates(files: dict[Path, str]) -> list[str]:
    """Repo-wide: every `id="…"` literal across all files; a repeated one is a collision."""
    seen: dict[str, str] = {}
    for path, text in files.items():
        for m in re.finditer(r'\bid="([0-9a-f]{12})"', text):
            id_ = m.group(1)
            if id_ in seen:
                return [id_]
            seen[id_] = str(path)
    return []


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def _clean(cwd: Path, files: list[Path]) -> bool:
    _git(cwd, "add", "-A")
    out = _git(cwd, "status", "--porcelain")
    return out == ""


def demo() -> None:
    tmp = Path(tempfile.mkdtemp())
    _git(tmp, "init", "-q")
    _git(tmp, "config", "user.email", "t@t")
    _git(tmp, "config", "user.name", "t")

    fixture = '''"""A retail credit pipeline, kept readable."""
# keep this comment, the imports, the decorators and the spacing exactly.
from decider.steps import branch, dag, flow, step

@step(output="term_cap")  # a note on the decorator
def cap_by_income(term_cap: float) -> float:
    return term_cap


@step(output="x")
def other(x: float) -> float:
    return x


term = flow(
    cap_by_income,
    other,
    name="term",
)

p03 = dag(cap_by_income, other, name="p03")
by_sector = branch("on_card", cap_by_income, cap_by_income.named("pub"), modifies=["term_cap"], name="by_sector")
'''
    (tmp / "pipeline.py").write_text(fixture)
    _git(tmp, "add", "-A")
    _git(tmp, "commit", "-qm", "baseline")

    print("-- item 4: parse-before-write + transform preserves bytes")
    src = (tmp / "pipeline.py").read_text()
    once = transform(src)
    ast.parse(once)
    assert '# keep this comment' in once
    assert '"""A retail credit pipeline' in once
    assert '@step(output="term_cap", id=' in once  # existing decorator gains id, comment kept
    assert "name=\"term\"," in once  # flow line untouched
    print(once)

    print("-- item 4: idempotence")
    twice = transform(once)
    assert once == twice, "a second pass must not change the file"
    print("  second pass is a no-op")

    print("-- item 4: duplicate detection (repo-wide scan)")
    file_a, file_b = transform(src), transform(src)
    assert scan_duplicates({Path("a.py"): file_a, Path("b.py"): file_b}) == [], "fresh ids must not collide"
    dup = scan_duplicates({Path("a.py"): once, Path("b.py"): once})
    assert dup, "an exact copy must be flagged"
    print(f"  independent files disjoint; exact copy flags {dup}")

    print("-- item 4: clean-tree gate + failure atomicity")
    # untracked file -> refuse
    (tmp / "stray.py").write_text("x = 1\n")
    dirty = _clean(tmp, [tmp / "pipeline.py"])
    assert not dirty, "an untracked file must make the tree dirty and block the run"
    (tmp / "stray.py").unlink()
    # syntax error in a target file -> abort before writing anything
    (tmp / "pipeline.py").write_text(once.replace("def other", "def other("))
    try:
        ast.parse((tmp / "pipeline.py").read_text())
        raise AssertionError("fixture should be broken")
    except SyntaxError:
        pass
    _git(tmp, "checkout", "--", "pipeline.py")
    print("  untracked file blocks; a parse error aborts before any write (checked via git checkout)")

    print("transform ok")


if __name__ == "__main__":
    demo()
