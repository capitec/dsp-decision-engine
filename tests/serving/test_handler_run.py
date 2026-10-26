"""RequestHandler.run is the in-process entry; process_fn is bytes -> run -> bytes."""
import json

import polars as pl
import pytest

from decikit import flow
from decikit.config import JsonFileStore
from decikit.serving import RequestHandler

from test_inputs import RECORD, build, due_day


def _handler(tmp_path, pipeline):
    store = JsonFileStore(basepath=str(tmp_path / "configs"))
    store.create_version({"params": {}})
    handler = RequestHandler(store, pipeline)
    handler.stage()
    handler.activate()
    return handler


def test_run_converts_string_dates_in_a_record(tmp_path):
    out = _handler(tmp_path, build).run(RECORD)
    assert out["due_day"] == 4


def test_run_converts_string_dates_in_a_frame(tmp_path):
    frame = pl.DataFrame({"decision_date": ["2026-01-31", "2026-02-01"], "tenure": [1, 2]})
    out = _handler(tmp_path, lambda: flow(due_day, name="d")).run(frame)
    assert out["due_day"].to_list() == [1, 3]


def test_process_fn_responses_are_byte_identical_to_the_async_handler(tmp_path):
    # Golden bytes captured from the coroutine version of process_fn.
    handler = _handler(tmp_path, build)
    assert handler.process_fn(json.dumps(RECORD).encode(), "application/json", "application/json").content == (
        b'{"decision_date":"2026-01-31","opened":["2026-01-01"],"history":[1.0,2.0],'
        b'"seen":"2026-01-31T09:30:00","extra":{"k":1},"bonus":null,"due_day":4}')
    handler = _handler(tmp_path, lambda: flow(due_day, name="d"))
    rows = json.dumps([{"decision_date": "2026-01-31", "tenure": 1},
                       {"decision_date": "2026-02-01", "tenure": 2}]).encode()
    assert handler.process_fn(rows, "application/json", "application/json").content == (
        b'[{"decision_date":"2026-01-31","tenure":1,"due_day":1},'
        b'{"decision_date":"2026-02-01","tenure":2,"due_day":3}]')
    assert handler.process_fn(rows, "text/csv", "application/json").content == (
        b"decision_date,tenure,due_day\n2026-01-31,1,1\n2026-02-01,2,3\n")


@pytest.mark.parametrize("hook", ["input_fn", "process_fn", "init_fn", "shutdown_fn", "output_fn"])
def test_an_async_hook_override_is_rejected_naming_the_method(hook):
    with pytest.raises(TypeError, match=rf"Bad\.{hook} is `async def`.*synchronous"):
        type("Bad", (RequestHandler,), {hook: _async_hook})


async def _async_hook(self, *args):
    pass


def test_a_subclass_run_of_its_own_leaves_process_fn_alone(tmp_path):
    class Replayer(RequestHandler):
        def run(self, request):
            return json.loads(self.process_fn(json.dumps(request).encode(), "application/json", "application/json").content)

    store = JsonFileStore(basepath=str(tmp_path / "configs"))
    store.create_version({"params": {}})
    handler = Replayer(store, build)
    handler.stage()
    handler.activate()
    assert handler.run(RECORD)["due_day"] == 4
