from decider._settings import get_settings


def test_settings_are_read_when_first_asked_for_and_cached(monkeypatch):
    monkeypatch.setenv("Decider_SERVE__PORT", "9001")
    assert get_settings().serve.port == 9001
    monkeypatch.setenv("Decider_SERVE__PORT", "9002")
    assert get_settings() is get_settings() and get_settings().serve.port == 9001
    get_settings.cache_clear()
    assert get_settings().serve.port == 9002


def test_checks_path_defaults_beside_datasets_path():
    s = get_settings()
    assert (s.datasets_path, s.checks_path) == ("${PROJECT_ROOT}/datasets", "${PROJECT_ROOT}/checks.yaml")
