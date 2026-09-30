"""The bridge relocation is a tested migration: `decider.debug_bridge` is the
canonical home, the old `decider_bridge` distribution and `tools/decider-bridge`
path are gone, and the extension's launch contract (`python -m decider.debug_bridge`)
still works."""
import importlib.util

from pathlib import Path

import decider.debug_bridge


def test_the_bridge_lives_under_decider_not_a_separate_distribution():
    # The relocated package is a submodule of `decider`, not a top-level module.
    assert decider.debug_bridge.__name__ == "decider.debug_bridge"
    assert importlib.util.find_spec("decider.debug_bridge") is not None
    # The old distribution name is gone; nothing imports it.
    assert importlib.util.find_spec("decider_bridge") is None


def test_the_old_tools_path_no_longer_exists():
    repo = Path(__file__).resolve().parents[2]
    assert not (repo / "tools" / "decider-bridge").exists()


def test_the_bridge_exposes_the_transport_helpers_under_decider():
    # The adapter helpers landed together, next to the session they adapt.
    from decider.debug_bridge.bridge import Bridge, main
    from decider.debug_bridge.loading import load_module
    from decider.debug_bridge.runs import tree_path
    from decider.debug_bridge.timeline import Timeline

    assert callable(Bridge) and callable(main) and callable(load_module) and callable(tree_path)
    assert callable(Timeline)


def test_the_launch_entry_point_is_python_m_decider_debug_bridge():
    # The extension's launch configuration is `python -m decider.debug_bridge`;
    # the entry module must exist. (Its import runs `main`, so assert on the
    # spec rather than importing it here; `test_stdio_protocol_round_trips`
    # already drives the real process.)
    assert importlib.util.find_spec("decider.debug_bridge.__main__") is not None
    assert importlib.util.find_spec("decider.debug_bridge.bridge") is not None
