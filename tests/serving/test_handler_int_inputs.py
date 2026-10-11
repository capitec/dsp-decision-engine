"""A JSON record or array declaring an `int` takes 7 or 7.0 and refuses 7.5, like `run` on a record or a frame."""
import json

import pytest

from decikit import flow
from decikit.exceptions import NotIntegralError

from test_handler_run import _handler


def count_double(count: int) -> int:
    return count * 2


def _post(handler, body):
    return handler.process_fn(json.dumps(body).encode(), "application/json", "application/json").content


def test_a_json_record_with_a_whole_float_for_an_int_is_accepted(tmp_path):
    handler = _handler(tmp_path, lambda: flow(count_double, name="d"))
    assert json.loads(_post(handler, {"count": 7.0}))["count_double"] == 14


@pytest.mark.parametrize("body", [{"count": 7.5}, [{"count": 1}, {"count": 7.5}]])
def test_a_json_record_or_array_with_a_fraction_for_an_int_is_refused(tmp_path, body):
    handler = _handler(tmp_path, lambda: flow(count_double, name="d"))
    with pytest.raises(NotIntegralError, match="input 'count'.*7.5"):
        _post(handler, body)
