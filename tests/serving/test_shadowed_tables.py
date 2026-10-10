"""Staging warns when a params document sets a table whose default rows come from a config document."""
import warnings

import pytest

from decikit import InterpretedEngine, ParamTable, ShadowedTableWarning, flow, param_table
from decikit._settings import get_settings
from decikit.config import JsonFileStore
from decikit.serving import RequestHandler

ROWS = [{"floor": 0, "rate": 0.2}, {"floor": 600, "rate": 0.1}]
COLUMNS = {"floor": int, "rate": float}


def seeded(score: int, rates: ParamTable = param_table(COLUMNS, default=ROWS, seeded_from="rate_card")) -> float:
    return float(len(rates.floor))


def seeded_shared(score: int, rates: ParamTable = param_table(
        COLUMNS, default=ROWS, seeded_from="rate_card", shared_key="card")) -> float:
    return float(len(rates.floor))


def allowed(score: int, rates: ParamTable = param_table(
        COLUMNS, default=ROWS, seeded_from="rate_card", allow_override=True)) -> float:
    return float(len(rates.floor))


def unseeded(score: int, rates: ParamTable = param_table(COLUMNS, default=ROWS)) -> float:
    return float(len(rates.floor))


def stage(tmp_path, step, params):
    store = JsonFileStore(basepath=str(tmp_path / "configs"))
    store.create_version({"params": params})
    handler = RequestHandler(store, flow(step, name="p"), engine=InterpretedEngine())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        handler.stage()
    return handler, [w for w in caught if issubclass(w.category, ShadowedTableWarning)]


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch):
    monkeypatch.delenv("DECIKIT_BUILD__WARN_SHADOWED_TABLES", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_staging_warns_when_params_set_a_seeded_table(tmp_path):
    _, caught = stage(tmp_path, seeded, {"p": {"seeded": {"rates": ROWS[:1]}}})
    [message] = [str(w.message) for w in caught]
    assert 'params key ["p"]["seeded"]["rates"] sets table `rates` of p/seeded' in message
    assert "config document `rate_card`" in message


def test_staging_warns_for_a_shared_seeded_table_and_names_the_shared_key(tmp_path):
    _, caught = stage(tmp_path, seeded_shared, {"shared": {"card": ROWS[:1]}})
    assert 'params key ["shared"]["card"]' in str(caught[0].message)


def test_staging_is_silent_when_params_leave_the_seeded_table_alone(tmp_path):
    assert stage(tmp_path, seeded, {})[1] == []
    assert stage(tmp_path, seeded, {"shared": {}})[1] == []


def test_staging_is_silent_for_a_table_that_is_not_seeded(tmp_path):
    assert stage(tmp_path, unseeded, {"p": {"unseeded": {"rates": ROWS[:1]}}})[1] == []


def test_a_run_time_params_override_stays_silent_and_applies(tmp_path):
    handler, _ = stage(tmp_path, seeded, {})
    handler.activate()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShadowedTableWarning)
        assert handler._active.executable.run({"score": 1}, {"p": {"seeded": {"rates": ROWS[:1]}}})["seeded"] == 1.0


def test_the_warning_can_be_ignored_by_category(tmp_path):
    store = JsonFileStore(basepath=str(tmp_path / "configs"))
    store.create_version({"params": {"p": {"seeded": {"rates": ROWS[:1]}}}})
    handler = RequestHandler(store, flow(seeded, name="p"), engine=InterpretedEngine())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=ShadowedTableWarning)
        handler.stage()


@pytest.mark.parametrize("value", ["false", "0"])
def test_the_warning_can_be_switched_off_by_a_setting(tmp_path, monkeypatch, value):
    monkeypatch.setenv("DECIKIT_BUILD__WARN_SHADOWED_TABLES", value)
    get_settings.cache_clear()
    assert stage(tmp_path, seeded, {"p": {"seeded": {"rates": ROWS[:1]}}})[1] == []


def test_the_warning_can_be_switched_off_for_one_table(tmp_path):
    assert stage(tmp_path, allowed, {"p": {"allowed": {"rates": ROWS[:1]}}})[1] == []


@pytest.mark.parametrize("params", [["not", "a", "mapping"], {"shared": None}])
def test_a_malformed_params_document_is_reported_by_validation_not_by_the_warning(tmp_path, monkeypatch, params):
    def failure(warn: str):
        monkeypatch.setenv("DECIKIT_BUILD__WARN_SHADOWED_TABLES", warn)
        get_settings.cache_clear()
        try:
            stage(tmp_path / warn, seeded_shared, params)
        except Exception as e:
            return type(e), str(e)
        return None

    assert failure("true") == failure("false")


def test_allow_override_without_seeded_from_is_refused():
    with pytest.raises(TypeError, match="allow_override only applies with seeded_from"):
        param_table(COLUMNS, default=ROWS, allow_override=True)
