"""`decider2 serve <pipeline.py>` — doc 02 §3.6 "keep it thin". Only
`load_pipeline`'s discovery convention is worth testing directly; actually
binding a socket is `serving/server.py`'s job and is exercised through
`tests/test_serving.py`'s ASGI-level tests instead of a real bind here.
"""
from __future__ import annotations

import textwrap

import click
import pytest

from decider2.cli import load_pipeline
from decider2.graph.pipeline import Pipeline


def test_loads_the_conventional_module_level_pipeline_name(tmp_path):
    path = tmp_path / "demo.py"
    path.write_text(textwrap.dedent(
        """
        from decider2 import flow, param

        def cap(term_cap: float, cap: float = param(48.0)) -> float:
            return min(term_cap, cap)

        pipeline = flow(cap)
        """
    ))
    loaded = load_pipeline(str(path))
    assert isinstance(loaded, Pipeline)


def test_an_explicit_attr_name_is_honoured(tmp_path):
    path = tmp_path / "demo.py"
    path.write_text(textwrap.dedent(
        """
        from decider2 import flow, param

        def cap(term_cap: float, cap: float = param(48.0)) -> float:
            return min(term_cap, cap)

        my_flow = flow(cap)
        """
    ))
    loaded = load_pipeline(f"{path}:my_flow")
    assert isinstance(loaded, Pipeline)


def test_no_pipeline_found_is_a_clean_cli_error(tmp_path):
    path = tmp_path / "empty.py"
    path.write_text("x = 1\n")
    with pytest.raises(click.ClickException):
        load_pipeline(str(path))


def test_a_missing_file_is_a_clean_cli_error(tmp_path):
    with pytest.raises(click.ClickException):
        load_pipeline(str(tmp_path / "does_not_exist.py"))


def test_a_dotted_module_path_also_works():
    loaded = load_pipeline("decider2.examples.flagship")
    assert isinstance(loaded, Pipeline)
