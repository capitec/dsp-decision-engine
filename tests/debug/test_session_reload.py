"""Hot reload: a session picks up an edited pipeline, re-running only from the first change."""
import os
import sys
import textwrap

import polars as pl
import pytest
from numba.core import event

from decider import flow, step
from decider.engine.debug import Edited, Error, Paused
from decider.engine.debug.hot import ModuleWatcher
from decider.testing import MODES

FRAME = pl.DataFrame({"net_income": [9200.0, 4100.0], "expenses": [3100.0, 1500.0], "instalment": [1200.0, 800.0]})


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def headroom(ratio: float) -> float:
    return ratio - 1.0


def halved(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment / 2.0


def ratio_rebuilt(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def buffer(headroom: float) -> float:
    return headroom * 0.5


@pytest.mark.parametrize("mode", MODES)
def test_reload_reruns_from_the_first_changed_step_in_every_mode(mode):
    s = flow(disposable_income, ratio, headroom).session(FRAME, mode=mode)
    s.resume()
    edited = flow(disposable_income, step(halved, name="ratio", output="ratio"), headroom)
    cp = s.reload(edited)
    assert (cp.origin.path, cp.when) == ("ratio", "before") or mode == "fused"
    assert Edited("replace", "ratio") in s.events and s.events[-1].reason == "edit"
    s.resume()
    assert s.output().equals(edited.run(FRAME))


def test_a_rebuilt_but_identical_pipeline_changes_nothing():
    s = flow(disposable_income, ratio, headroom).session(FRAME)
    s.resume()
    n = len(s.events)
    # New function object, same content: nothing to do.
    assert s.reload(flow(disposable_income, step(ratio_rebuilt, name="ratio", output="ratio"), headroom)) is None
    assert len(s.events) == n


def test_inserting_and_deleting_steps_are_add_and_delete_edits():
    s = flow(disposable_income, ratio, headroom).session(FRAME)
    s.resume()
    s.reload(flow(disposable_income, ratio, headroom, buffer))
    assert s.events[-2:] == [Edited("add", "buffer"), Paused(s.current.origin, "before", "edit")]
    s.resume()
    assert s.output()["buffer"].to_list() == [((6100 / 1200) - 1) / 2, ((2600 / 800) - 1) / 2]
    s.reload(flow(disposable_income, ratio, headroom))
    assert Edited("delete", "buffer") in s.events
    assert "buffer" not in s.output().columns


def test_a_changed_param_default_counts_as_a_change():
    from decider import param

    def cap(ratio: float, most: float = param(2.0)) -> float:
        return min(ratio, most)

    def cap_lower(ratio: float, most: float = param(1.0)) -> float:
        return min(ratio, most)

    s = flow(disposable_income, ratio, cap).session(FRAME)
    s.resume()
    s.reload(flow(disposable_income, ratio, step(cap_lower, name="cap", output="cap")))
    s.resume()
    assert s.output()["cap"].to_list() == [1.0, 1.0]


def test_a_reload_that_doesnt_wire_changes_nothing():
    s = flow(disposable_income, ratio, headroom).session(FRAME)
    s.resume()
    with pytest.raises(ValueError):
        s.reload(flow(disposable_income, headroom))
    assert s.output().equals(flow(disposable_income, ratio, headroom).run(FRAME))


NOTEBOOK = [
    "import polars as pl\nfrom decider import flow",
    "def helper(x):\n    return x * 1.0",
    "def ratio(disposable_income: float, instalment: float) -> float:\n    return helper(disposable_income / instalment)",
    "def disposable_income(net_income: float, expenses: float) -> float:\n    return net_income - expenses",
]


def test_a_notebook_cell_redefining_a_helper_is_picked_up_by_a_watch_that_rebuilds_the_flow():
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    shell.user_ns["FRAME"] = FRAME
    try:
        for cell in NOTEBOOK:
            shell.run_cell(cell)
        # The lambda looks its names up in the notebook when called, so redefining a function is enough.
        shell.run_cell("s = flow(disposable_income, ratio).session(FRAME)\ns.resume()\n"
                       "s.watch(lambda: flow(disposable_income, ratio))")
        s = shell.user_ns["s"]
        shell.run_cell("def helper(x):\n    return x * 10.0")
        assert Edited("replace", "ratio") in s.events
        shell.run_cell("s.resume()")
        assert s.output()["ratio"].to_list() == [6100 / 1200 * 10, 2600 / 800 * 10]
        # A cell that breaks the pipeline: logged, and the session keeps what it had.
        shell.run_cell("del ratio")
        assert isinstance(s.events[-1], Error)
        assert s.output()["ratio"].to_list() == [6100 / 1200 * 10, 2600 / 800 * 10]
    finally:
        shell.events.callbacks["post_run_cell"].clear()
        InteractiveShell.clear_instance()


PIPELINE = """
from decider import flow
from {pkg}.features import ratio

def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses

pipeline = flow(disposable_income, ratio)
"""


def _write(path, text):
    path.write_text(textwrap.dedent(text))
    # Well past the previous stamp, whatever the filesystem's clock granularity.
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


@pytest.mark.parametrize("mode", ["interpreted", "stepped"])
def test_editing_an_imported_module_on_disk_reloads_the_pipeline_without_stale_references(tmp_path, monkeypatch, mode):
    pkg = f"hot_{mode}"
    root = tmp_path / pkg
    root.mkdir()
    (root / "__init__.py").write_text("")
    _write(root / "features.py", "def ratio(disposable_income: float, instalment: float) -> float:\n"
                                 "    return disposable_income / instalment\n")
    _write(root / "pipeline.py", PIPELINE.format(pkg=pkg))
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        watcher = ModuleWatcher(f"{pkg}.pipeline:pipeline")
        s = watcher.get().session(FRAME, mode=mode)
        s.resume()
        assert watcher.poll() is None
        _write(root / "features.py", "def ratio(disposable_income: float, instalment: float) -> float:\n"
                                     "    return disposable_income / instalment / 4.0\n")
        new = watcher.poll()
        with event.install_recorder("numba:compile") as compiled:
            s.reload(new)
        # Only the edited function compiles; the re-imported, unchanged one is shared by content.
        assert {e.data["dispatcher"].py_func.__qualname__ for _, e in compiled.buffer} <= {"ratio"}
        assert s.events[-2] == Edited("replace", "ratio")
        s.resume()
        assert s.output()["ratio"].to_list() == [6100 / 1200 / 4, 2600 / 800 / 4]
        _write(root / "features.py", "def ratio(:\n")
        with pytest.raises(SyntaxError):
            watcher.poll()
        # The broken edit leaves the previous modules importable and in place.
        assert sys.modules[f"{pkg}.pipeline"].pipeline is new
    finally:
        for name in [n for n in sys.modules if n.startswith(pkg)]:
            del sys.modules[name]

