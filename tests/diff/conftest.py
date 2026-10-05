import subprocess

import pytest

PIPELINE = '''
from decider import flow


def ratio(income: float, debt: float) -> float:
    return debt / income * {factor}


def build():
    return flow(ratio, name="credit")
'''


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repo whose `shop.pipeline:build` has two commits: factor 1.0, then factor 2.0."""
    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path, check=True,
                       capture_output=True)

    git("init", "-q")
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    for factor in ("1.0", "2.0"):
        (tmp_path / "shop" / "pipeline.py").write_text(PIPELINE.format(factor=factor))
        git("add", "-A")
        git("commit", "-q", "-m", f"factor {factor}")
    monkeypatch.chdir(tmp_path)
    return tmp_path
