"""A JSON array of requests answers like the same requests sent one by one."""
import datetime as dt
import json

import pytest
from typing_extensions import NotRequired, TypedDict

from decider import flow
from decider.ops import Columnar, Struct
from decider.config import JsonFileStore
from decider.serving import RequestHandler


class Account(TypedDict):
    balance: float
    limit: NotRequired[float]
    note: NotRequired[str | None]
    opened: NotRequired[dt.date]
    tags: NotRequired[list[str]]


class Plain(TypedDict):
    balance: float
    limit: NotRequired[float]


def shape(accounts: list[Account]) -> str:
    return json.dumps([sorted(a) for a in accounts])


def shape_plain(plain: list[Plain]) -> str:
    return json.dumps([sorted(a) for a in plain])


def maybe(accounts: list[Account], bonus: float | None, kind: str) -> str:
    return f"{kind}:{bonus}:{len(accounts)}"


def _handler(tmp_path, *steps, sample=None):
    store = JsonFileStore(basepath=str(tmp_path / "configs"))
    store.create_version({"params": {}})
    if sample is not None:
        (tmp_path / "sample.json").write_text(json.dumps(sample))
    handler = RequestHandler(store, lambda: flow(*steps, name="f"),
                             sample_request=None if sample is None else str(tmp_path / "sample.json"))
    handler.stage()
    handler.activate()
    return handler


def _send(handler, body) -> list | dict:
    return json.loads(handler.process_fn(json.dumps(body).encode(), "application/json", "application/json").content)


DATED = {"accounts": [{"balance": 1.0, "opened": "2026-01-01"}, {"balance": 2.0, "limit": 5.0, "tags": ["a"]}]}
PLAIN = {"plain": [{"balance": 1.0}, {"balance": 2.0, "limit": 5.0}]}


@pytest.mark.parametrize("step, request_", [(shape, DATED), (shape_plain, PLAIN)])
def test_a_not_required_key_left_out_is_absent_for_the_step_in_an_array_as_in_a_record(tmp_path, step, request_):
    handler = _handler(tmp_path, step)
    out = step.__name__
    single = _send(handler, request_)[out]
    assert _send(handler, [request_, request_]) and [r[out] for r in _send(handler, [request_, request_])] == [single, single]
    assert "limit" not in json.loads(single)[0]


def test_an_array_mixing_records_with_and_without_a_not_required_list_matches_records(tmp_path):
    handler = _handler(tmp_path, shape)
    rows = [{"accounts": [{"balance": 1.0}]}, {"accounts": [{"balance": 2.0, "tags": ["x"], "limit": 3.0}]}]
    assert [r["shape"] for r in _send(handler, rows)] == [_send(handler, r)["shape"] for r in rows]


def test_a_required_key_sent_as_null_in_an_array_is_refused_naming_row_and_field(tmp_path):
    handler = _handler(tmp_path, shape)
    rows = [{"accounts": [{"balance": 1.0}]}, {"accounts": [{"balance": None}]}]
    with pytest.raises(Exception, match=r"row 1 field 0.balance"):
        _send(handler, rows)


def test_a_wrong_type_that_is_not_a_date_is_not_blamed_on_dates(tmp_path):
    handler = _handler(tmp_path, shape)
    rows = [{"accounts": [{"balance": "lots"}]}, {"accounts": [{"balance": 1.0}]}]
    with pytest.raises(Exception) as e:
        _send(handler, rows)
    assert "balance" in str(e.value) and "ISO" not in str(e.value) and "dates" not in str(e.value)


def test_a_bad_date_in_an_array_still_says_to_send_iso_strings(tmp_path):
    handler = _handler(tmp_path, shape)
    with pytest.raises(Exception, match="ISO strings"):
        _send(handler, [{"accounts": [{"balance": 1.0, "opened": "31/01/2026"}]}] * 2)


class Priced(TypedDict):
    price: float
    qty: NotRequired[float]


def priced_total(items: Columnar[Priced]) -> float:
    t = 0.0
    for j in range(len(items.price)):
        t += items.price[j] * items.qty[j]
    return t


class Applicant(TypedDict):
    income: float
    bonus: NotRequired[float]


def income(applicant: Struct[Applicant] | None) -> float:
    return -1.0 if applicant is None else applicant["income"]


@pytest.mark.parametrize("step, rows", [
    (priced_total, [{"items": [{"price": 2.0, "qty": 3.0}]}, {"items": [{"price": 5.0, "qty": 1.0}]}]),
    (income, [{"applicant": {"income": 1.0, "bonus": 2.0}}, {"applicant": None}, {"applicant": {"income": 3.0}}]),
])
def test_an_array_for_a_columnar_or_optional_struct_input_with_a_not_required_key_matches_records(tmp_path, step, rows):
    handler = _handler(tmp_path, step, sample=rows[0])
    out = step.__name__
    assert [r[out] for r in _send(handler, rows)] == [_send(handler, r)[out] for r in rows]


def test_an_optional_input_and_a_nullable_key_keep_their_null_in_an_array_as_in_a_record(tmp_path):
    handler = _handler(tmp_path, maybe)
    body = {"accounts": [{"balance": 1.0, "note": None}, {"balance": 2.0, "note": "n"}], "bonus": None, "kind": "k"}
    assert _send(handler, [body, {**body, "bonus": 2.0}]) and [r["maybe"] for r in _send(handler, [body, {**body, "bonus": 2.0}])] == [
        _send(handler, body)["maybe"], _send(handler, {**body, "bonus": 2.0})["maybe"]]
