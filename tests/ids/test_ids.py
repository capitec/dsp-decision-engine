"""Durable id generation: byte-splice, safety gates, and evolution scenarios."""
import ast
import re
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from decider import engine
from decider.cli import cli
from decider.engine.ir.nodes import iter_nodes
from decider.ids import IdError, add_ids, generate
from decider.steps import flow, step

PIPELINE = '''"""A retail credit pipeline, kept readable."""
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


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout


def _commit(repo: Path, message: str = "baseline") -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    return tmp_path


# ---- byte-splice (pure) -----------------------------------------------------

def test_add_ids_preserves_comments_and_formatting():
    text, changes = add_ids(PIPELINE)
    ast.parse(text)
    assert '# keep this comment' in text
    assert '"""A retail credit pipeline' in text
    assert '@step(output="term_cap", id=' in text  # existing decorator gains id, comment kept
    assert 'name="term",\n    id="' in text  # flow gains an id on its own line
    assert len(changes) == 5 and text.count('id="') == 5


def test_add_ids_handles_a_bare_decorator_and_a_trailing_paren():
    src = '@step\ndef a(x: float) -> float:\n    return x\n\nterm = flow(\n    a,\n    name="t")'
    text, changes = add_ids(src)
    ast.parse(text)
    assert '@step(id="' in text
    assert 'name="t",\n    id="' in text  # trailing paren case is reformatted to paren-on-own-line
    assert text.endswith(')')


def test_add_ids_is_idempotent():
    once, _ = add_ids(PIPELINE)
    twice, changes = add_ids(once)
    assert twice == once and changes == []


def test_add_ids_leaves_anonymous_flows_and_existing_ids_untouched():
    src = 'pipeline = flow(a, b)\nwrapped = step(c, id="0123abcdef45")\n'
    text, changes = add_ids(src)
    assert text == src and changes == []


# ---- flow id + step id is a global reference --------------------------------

def test_flow_and_step_ids_form_a_global_reference():
    @step(output="x", id="aaaaaaaaaaaa")
    def one(x: float) -> float:
        return x

    root = flow(flow(one, name="inner", id="bbbbbbbbbbbb"), name="root", id="cccccccccccc")
    by_path = {n.origin.path: n.origin for n in iter_nodes(engine.to_ir(root))}
    assert by_path["root"].id == "cccccccccccc"
    assert by_path["root/inner"].id == "bbbbbbbbbbbb"
    assert by_path["root/inner/one"].id == "aaaaaaaaaaaa"


# ---- generator safety (git fixture) -----------------------------------------

def test_generate_adds_ids_to_a_clean_tree(repo: Path):
    (repo / "pipeline.py").write_text(PIPELINE)
    _commit(repo)
    report = generate(repo)
    text = (repo / "pipeline.py").read_text()
    ast.parse(text)
    assert report.files == 1 and len(report.changes) == 5
    assert text.count('id="') == 5


def test_generate_refuses_a_dirty_tree(repo: Path):
    (repo / "pipeline.py").write_text(PIPELINE)
    _commit(repo)
    (repo / "stray.py").write_text("x = 1\n")
    with pytest.raises(IdError, match="not clean"):
        generate(repo)


def test_generate_refuses_outside_a_git_repository(tmp_path: Path):
    (tmp_path / "pipeline.py").write_text(PIPELINE)
    with pytest.raises(IdError, match="not inside a git repository"):
        generate(tmp_path)


def test_generate_aborts_on_a_syntax_error_without_writing(repo: Path):
    good = repo / "good.py"
    good.write_text('@step(output="x")\ndef a(x: float) -> float:\n    return x\n')
    (repo / "bad.py").write_text("def broken(:\n")
    _commit(repo)
    before = good.read_text()
    with pytest.raises(IdError, match="syntax error"):
        generate(repo)
    assert good.read_text() == before


def test_generate_rejects_a_duplicate_id(repo: Path):
    (repo / "pipeline.py").write_text(
        '@step(output="x", id="aaaaaaaaaaaa")\ndef a(x: float) -> float:\n    return x\n\n'
        '@step(output="y", id="aaaaaaaaaaaa")\ndef b(y: float) -> float:\n    return y\n'
    )
    _commit(repo)
    with pytest.raises(IdError, match="duplicate id"):
        generate(repo)


def test_generate_fix_reassigns_the_later_copy(repo: Path):
    (repo / "pipeline.py").write_text(
        '@step(output="x", id="aaaaaaaaaaaa")\ndef a(x: float) -> float:\n    return x\n\n'
        '@step(output="y", id="aaaaaaaaaaaa")\ndef b(y: float) -> float:\n    return y\n'
    )
    _commit(repo)
    report = generate(repo, fix=True)
    text = (repo / "pipeline.py").read_text()
    assert text.count('id="aaaaaaaaaaaa"') == 1  # the first copy keeps it
    assert len(report.changes) == 1
    assert len(set(re.findall(r'id="([0-9a-f]{12})"', text))) == 2


def test_generate_rejects_a_malformed_id(repo: Path):
    (repo / "pipeline.py").write_text('@step(output="x", id="short")\ndef a(x: float) -> float:\n    return x\n')
    _commit(repo)
    with pytest.raises(IdError, match="12 lowercase hex"):
        generate(repo)


def test_generate_rejects_an_invalid_step_name(repo: Path):
    (repo / "pipeline.py").write_text('pipe = flow(a, name="bad/name")\n')
    _commit(repo)
    with pytest.raises(IdError, match="must be a non-empty string"):
        generate(repo)


def test_generate_check_reports_without_writing(repo: Path):
    (repo / "pipeline.py").write_text(PIPELINE)
    _commit(repo)
    before = (repo / "pipeline.py").read_text()
    report = generate(repo, check=True)
    assert len(report.changes) == 5 and report.files == 1
    assert (repo / "pipeline.py").read_text() == before


# ---- evolution scenarios (rename / extract / reorder / user-supplied id) ----

def test_user_supplied_ids_are_preserved(repo: Path):
    (repo / "pipeline.py").write_text(
        '@step(output="x", id="0123abcdef45")\ndef a(x: float) -> float:\n    return x\n\n'
        'pipe = flow(a, name="p")\n'
    )
    _commit(repo)
    report = generate(repo)
    text = (repo / "pipeline.py").read_text()
    assert 'id="0123abcdef45"' in text
    assert [c[2] for c in report.changes] == ["p"]


def test_committed_ids_survive_a_rename(repo: Path):
    (repo / "pipeline.py").write_text(
        '@step(output="x")\ndef a(x: float) -> float:\n    return x\n\npipe = flow(a, name="p")\n'
    )
    _commit(repo)
    generate(repo)
    _commit(repo, "ids")
    text = (repo / "pipeline.py").read_text()
    id_ = text.split('id="')[1].split('"')[0]
    (repo / "pipeline.py").write_text(text.replace("def a", "def a_renamed"))
    _commit(repo, "rename")
    assert generate(repo).changes == []
    assert f'id="{id_}"' in (repo / "pipeline.py").read_text()


def test_committed_ids_survive_reordering(repo: Path):
    (repo / "pipeline.py").write_text(
        'from decider.steps import flow, step\n\n'
        '@step(output="x")\ndef a(x: float) -> float:\n    return x\n\n'
        '@step(output="y")\ndef b(y: float) -> float:\n    return y\n\n'
        'pipe = flow(a, b, name="p")\n'
    )
    _commit(repo)
    generate(repo)
    _commit(repo, "ids")
    text = (repo / "pipeline.py").read_text()
    (repo / "pipeline.py").write_text(text.replace("flow(a, b,", "flow(b, a,"))
    _commit(repo, "reorder")
    assert generate(repo).changes == []


def test_committed_ids_survive_extraction(repo: Path):
    (repo / "pipeline.py").write_text(
        'from decider.steps import flow, step\n\n'
        '@step(output="x")\ndef a(x: float) -> float:\n    return x\n\n'
        'pipe = flow(a, name="p")\n'
    )
    _commit(repo)
    generate(repo)
    _commit(repo, "ids")
    text = (repo / "pipeline.py").read_text()
    decl = text.split("pipe =")[0]
    id_ = decl.split('id="')[1].split('"')[0]
    (repo / "helpers.py").write_text(decl)
    (repo / "pipeline.py").write_text('from helpers import a\n\npipe = flow(a, name="p", id="bbbbbbbbbbbb")\n')
    _commit(repo, "extract")
    assert generate(repo).changes == []
    assert f'id="{id_}"' in (repo / "helpers.py").read_text()


def test_cli_ids_adds_ids(repo: Path, monkeypatch):
    (repo / "pipeline.py").write_text(PIPELINE)
    _commit(repo)
    monkeypatch.chdir(repo)
    result = CliRunner().invoke(cli, ["ids"])
    assert result.exit_code == 0, result.output
    assert "added 5 id(s) in 1 file(s)" in result.output
