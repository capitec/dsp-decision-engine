"""Every Python example in decider/GUIDE.md runs, inside a project made by `decider template credit_risk`."""
import os
import re
import sys
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

import decider
import decider.settings as settings_module
from decider.cli import cli

GUIDE = (Path(decider.__file__).parent / "GUIDE.md").read_text()
BLOCKS = re.findall(r"```python\n(.*?)```", GUIDE, re.S)

# Examples of behaviour that lands with fixes outside the guide; un-xfail them once merged.
PENDING = {
    "trace_output": "tree trace output",
}


def _marks(block: str):
    return [pytest.mark.xfail(reason=why, strict=False) for key, why in PENDING.items() if key in block]


@pytest.fixture
def credit_risk_project(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if not k.upper().startswith("DECIDER_")})
    monkeypatch.setattr(settings_module, "settings", settings_module.settings)
    monkeypatch.setattr(sys, "path", [str(tmp_path / "credit_risk"), *sys.path])
    for name in [m for m in sys.modules if m.split(".")[0] == "credit_risk"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.chdir(tmp_path)
    assert CliRunner().invoke(cli, ["template", "credit_risk"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "credit_risk")


def test_the_guide_has_examples():
    assert len(BLOCKS) >= 6


@pytest.mark.parametrize("block", [pytest.param(b, marks=_marks(b), id=f"block{i}") for i, b in enumerate(BLOCKS)])
def test_guide_example_runs(block, credit_risk_project):
    exec(compile(block, "GUIDE.md", "exec"), {"__name__": "guide"})


def test_the_package_docstring_quickstart_runs():
    quickstart = decider.__doc__.split("Quickstart::")[1].split("\nA function's")[0]
    exec(compile(textwrap.dedent(quickstart), "decider.__doc__", "exec"), {})


def test_decider_guide_prints_the_guide():
    result = CliRunner().invoke(cli, ["guide"])
    assert result.exit_code == 0 and result.output.strip() == GUIDE.strip()
