"""The bridge relocation is a tested migration: `decikit._tooling.bridge` is the
canonical home, the old `decikit_bridge` distribution and `tools/decikit-bridge`
path are gone, and the extension's launch contract (`python -m decikit._tooling.bridge`)
still works."""
import importlib.util

from pathlib import Path

import decikit._tooling.bridge


def test_the_bridge_lives_under_decikit_not_a_separate_distribution():
    # The relocated package is a submodule of `decikit`, not a top-level module.
    assert decikit._tooling.bridge.__name__ == "decikit._tooling.bridge"
    assert importlib.util.find_spec("decikit._tooling.bridge") is not None
    # The old distribution name is gone; nothing imports it.
    assert importlib.util.find_spec("decikit_bridge") is None


def test_the_old_tools_path_no_longer_exists():
    # The tracked source is gone; untracked build leftovers (a stale `.venv`,
    # `__pycache__`) may keep the directory itself on disk, so assert on source.
    repo = Path(__file__).resolve().parents[2]
    old = repo / "tools" / "decikit-bridge" / "decikit_bridge"
    assert not (old / "bridge.py").exists()
    assert not (old / "__init__.py").exists()


def test_the_tooling_lives_under_one_private_package():
    for old in ("decikit.cli", "decikit.mcp", "decikit._bridge", "decikit._tools"):
        assert importlib.util.find_spec(old) is None, old


def test_the_bridge_exposes_the_transport_helpers_under_decikit():
    # The adapter helpers landed together, next to the session they adapt.
    from decikit._tooling.bridge.bridge import Bridge, main
    from decikit._tooling.bridge.loading import load_module
    from decikit._tooling.bridge.runs import tree_path
    from decikit._tooling.bridge.timeline import Timeline

    assert callable(Bridge) and callable(main) and callable(load_module) and callable(tree_path)
    assert callable(Timeline)


def test_the_launch_entry_point_is_python_m_decikit_bridge():
    # The extension's launch configuration is `python -m decikit._tooling.bridge`;
    # the entry module must exist. (Its import runs `main`, so assert on the
    # spec rather than importing it here; `test_stdio_protocol_round_trips`
    # already drives the real process.)
    assert importlib.util.find_spec("decikit._tooling.bridge.__main__") is not None
    assert importlib.util.find_spec("decikit._tooling.bridge.bridge") is not None
