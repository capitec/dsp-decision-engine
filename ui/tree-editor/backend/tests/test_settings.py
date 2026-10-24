from dsp_decikit.settings import AppSettings


def test_overriding_one_cookie_field_keeps_the_other_defaults(monkeypatch):
    monkeypatch.setenv("DSP_AUTH__STATUS_COOKIE__SECURE", "false")
    monkeypatch.setenv("DSP_AUTH__COOKIE__SECURE", "false")
    auth = AppSettings().auth
    assert (auth.status_cookie.key, auth.status_cookie.httponly, auth.status_cookie.secure) == ("session_status", False, False)
    assert (auth.cookie.key, auth.cookie.httponly, auth.cookie.secure) == ("session_id", True, False)
