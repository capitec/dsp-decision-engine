"""Request handling for the credit-core demo.

`decider build`/`decider serve` warm every kernel with a synthetic record
before serving (`decider.serving.handler._warm`). That synthesiser only
knows `bool`/`int`/`str`/`bytes` (`decider/serving/handler.py`'s `_DUMMY`
dict) and falls back to the float `1.0` for every other type -- including
`list`, `dict` and `datetime.date`. This pipeline's top-level inputs
include `decision_date: date`, `bureau_accounts: list[dict]` and
`variable_pay_history: list[float]` (00 §7.3 makes `decision_date`
mandatory on every capability; ragged account lists are 00 §6.4's own
"Hard part"), so the built-in warm-up crashes before any request is ever
served:

    TypeError: 'float' object is not iterable
      File ".../credit_core/income.py", line 125, in income_variability_ratio
        values = [v for v in history if v is not None]

This is not something a `Handler.*_fn` override reaches -- `stage()` calls
the module-level `_warm` directly, with no extension point -- so this
module replaces it, at import time, with one that warms using this
project's own `sample_request.json` instead of a synthesised record. A
real, complete, representative record is arguably a better warm-up input
than a synthetic one regardless of the bug (00 §9's "no per-request
compilation" requirement, faithfully warmed). See NOTES.md "Framework
friction" for the full writeup and repro.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import decider.serving.handler as _handler
from decider.serving.handler import RequestHandler

_SAMPLE_REQUEST_PATH = Path(__file__).parent / "sample_request.json"


def _typed_sample_record() -> dict:
    record = json.loads(_SAMPLE_REQUEST_PATH.read_text())
    record["decision_date"] = datetime.date.fromisoformat(record["decision_date"])
    for account in record.get("bureau_accounts", []) + record.get("internal_accounts", []):
        if account.get("opened_date"):
            account["opened_date"] = datetime.date.fromisoformat(account["opened_date"])
    return record


def _warm_with_sample_record(exe, params) -> None:
    record = _typed_sample_record()
    exe.score(record, params)
    import polars as pl
    exe.run(pl.DataFrame([record]), params)


_handler._warm = _warm_with_sample_record


class Handler(RequestHandler):
    pass
