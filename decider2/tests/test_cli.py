"""`decider2 serve <pipeline.py>` — doc 02 §3.6 "keep it thin". Only
`load_pipeline`'s discovery convention is worth testing directly; actually
binding a socket is `serving/server.py`'s job and is exercised through
`tests/test_serving.py`'s ASGI-level tests instead of a real bind here.
"""
from __future__ import annotations

import os
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


# `decider2 build [--verify]` — a thin wrapper over `Pipeline.precompile()`
# and `decider2.testing.assert_no_compilation_after_warmup` (doc 05 §8).

def _write_flow(tmp_path):
    path = tmp_path / "demo.py"
    path.write_text(textwrap.dedent(
        """
        from decider2 import flow, param

        def cap(term_cap: float, cap: float = param(48.0)) -> float:
            return min(term_cap, cap)

        pipeline = flow(cap)
        """
    ))
    return path


def test_build_precompiles_and_reports(tmp_path):
    from click.testing import CliRunner
    from decider2.cli import cli

    result = CliRunner().invoke(cli, ["build", str(_write_flow(tmp_path))])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("precompile: apply ")
    assert "compile events" in result.output and "verify" not in result.output


def test_build_verify_is_the_zero_compilation_release_gate(tmp_path):
    from click.testing import CliRunner
    import decider2._arrow as arrow
    from decider2.cli import cli

    # `build --verify` gates a release on the compiled shim being loadable, so
    # it cannot pass without it. Skip where the extension was never built (no C
    # toolchain or headers) rather than fail; DECIDER2_REQUIRE_SHIM=1, which CI
    # sets, turns that back into an error. The sibling test below covers the
    # unavailable case on purpose, by monkeypatching.
    if arrow.diagnose()["shim"] != "loaded" and not os.environ.get("DECIDER2_REQUIRE_SHIM"):
        pytest.skip("compiled shim not built here; set DECIDER2_REQUIRE_SHIM=1 to require it")

    result = CliRunner().invoke(cli, ["build", "--verify", str(_write_flow(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "shim: loaded (" in result.output and "nanoarrow 0.9.0" in result.output
    assert result.output.rstrip().endswith("verify: 0 compilations after warm-up")


def test_build_verify_fails_cleanly_when_the_shim_is_unavailable(tmp_path, monkeypatch):
    from click.testing import CliRunner
    import decider2._arrow as arrow
    from decider2.cli import cli

    monkeypatch.setattr(arrow, "diagnose", lambda: {"shim": "unavailable", "error": "no shim here"})
    result = CliRunner().invoke(cli, ["build", "--verify", str(_write_flow(tmp_path))])
    assert result.exit_code != 0
    assert "no shim here" in result.output
