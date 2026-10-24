"""Sign-in sessions and OIDC logins across two backend processes that share one Postgres, as behind a load
balancer without sticky routing.

Needs a migrated Postgres (the e2e stack's: `e2e/stack.sh up`) and runs only when TREE_EDITOR_DB=1, with
DSP_DB__PORT pointing at it. Each node pair runs once per cache provider: sessions must not depend on it.
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from base64 import urlsafe_b64encode
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from jose import jwt

pytestmark = pytest.mark.skipif(os.environ.get("TREE_EDITOR_DB") != "1",
                                reason="needs the e2e stack's Postgres (TREE_EDITOR_DB=1)")

BACKEND = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("SESSIONS_TEST_PORT", "9100"))
ISSUER = f"http://127.0.0.1:{PORT + 99}"
CLIENT_ID = "tree-editor"
SECRET = "test-signing-secret-of-at-least-32-bytes"
ROLLING_S = 4  # a session idle this long expires
PROBE = "/api/v1/user/info"

CACHES = {
    "in-memory": None,
    "persisted-cache": {"provider": "persisted-cache", "cache": {"provider": "in-memory"}},
}


class FakeIdentityProvider(BaseHTTPRequestHandler):
    """Discovery, keys and a token endpoint that signs an id_token for whatever code it is given."""

    def do_GET(self):
        if self.path == "/.well-known/openid-configuration":
            self._json({
                "issuer": ISSUER, "authorization_endpoint": f"{ISSUER}/authorize",
                "token_endpoint": f"{ISSUER}/token", "jwks_uri": f"{ISSUER}/jwks",
                "end_session_endpoint": f"{ISSUER}/logout", "userinfo_endpoint": f"{ISSUER}/userinfo",
                "id_token_signing_alg_values_supported": ["HS256"], "response_types_supported": ["code"],
                "scopes_supported": ["openid", "profile", "email", "groups"],
            })
        elif self.path == "/jwks":
            k = urlsafe_b64encode(SECRET.encode()).rstrip(b"=").decode()
            self._json({"keys": [{"kty": "oct", "kid": "k1", "alg": "HS256", "k": k}]})
        else:
            self.send_error(404)

    def do_POST(self):
        form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
        now = int(time.time())
        claims = {"iss": ISSUER, "aud": CLIENT_ID, "sub": form["code"][0], "iat": now, "exp": now + 300,
                  "email": f"{form['code'][0]}@example.com", "name": "Oidc User", "groups": ["oidc-team"]}
        self._json({"id_token": jwt.encode(claims, SECRET, algorithm="HS256", headers={"kid": "k1"})})

    def _json(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def identity_provider():
    server = ThreadingHTTPServer(("127.0.0.1", PORT + 99), FakeIdentityProvider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield
    server.shutdown()


def node_env(cache) -> dict:
    oidc = {"enabled": True, "providers": {"fake": {
        "client_id": CLIENT_ID, "client_secret": "client-secret", "authority": ISSUER,
        "group_remapping_rules": [{"find": "^(.*)$", "replace": "\\1"}],
    }}}
    env = {**os.environ, "DSP_AUTH__COOKIE__SECURE": "false", "DSP_AUTH__STATUS_COOKIE__SECURE": "false",
           "DSP_AUTH__SESSION_ROLLING_DURATION_S": str(ROLLING_S),
           "DSP_AUTH__SESSION_RENEWAL_THRESHOLD_S": str(ROLLING_S - 1), "DSP_AUTH__OIDC": json.dumps(oidc)}
    env.pop("DSP_CACHE", None)
    if cache:
        env["DSP_CACHE"] = json.dumps(cache)
    return env


@pytest.fixture(scope="module", params=list(CACHES))
def nodes(request, identity_provider, tmp_path_factory):
    """Two backend processes, A and B, as base URLs."""
    first = PORT + 2 * list(CACHES).index(request.param)
    logs = tmp_path_factory.mktemp("nodes")
    procs, urls = [], []
    for port in (first, first + 1):
        log = open(logs / f"node-{port}.log", "w")
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "dsp_decikit.asgi:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=BACKEND, env=node_env(CACHES[request.param]), stdout=log, stderr=subprocess.STDOUT))
        urls.append(f"http://127.0.0.1:{port}")
    try:
        for url, proc in zip(urls, procs):
            for _ in range(120):
                assert proc.poll() is None, (logs / f"node-{url.rsplit(':', 1)[1]}.log").read_text()[-3000:]
                try:
                    if httpx.get(url + "/api/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.5)
            else:
                pytest.fail(f"{url} did not start")
        yield urls
    finally:
        for proc in procs:
            proc.terminate()
            proc.wait(10)


def sign_up(node: str) -> str:
    """Register a fresh user on `node`; returns the session cookie."""
    response = httpx.post(node + "/api/auth/basic/register",
                          data={"email": f"{uuid.uuid4().hex[:12]}@example.com", "password": "pw-123456", "name": "N"})
    assert response.status_code == 302, response.text
    return response.cookies["session_id"]


def accepted(node: str, session: str) -> bool:
    return httpx.get(node + PROBE, cookies={"session_id": session}).status_code == 200


def sign_out(node: str, session: str) -> None:
    assert httpx.get(node + "/api/auth/logout", cookies={"session_id": session}).status_code == 302


def test_a_session_made_on_one_node_is_accepted_on_the_other(nodes):
    a, b = nodes
    session = sign_up(a)

    assert accepted(b, session)
    assert accepted(a, session)


def test_signing_out_on_one_node_is_refused_on_the_other(nodes):
    a, b = nodes
    session = sign_up(a)
    assert accepted(b, session)

    sign_out(a, session)

    assert not accepted(b, session)
    assert not accepted(a, session)


def test_a_session_renewed_on_one_node_is_not_expired_by_the_other(nodes):
    a, b = nodes
    session = sign_up(a)
    assert accepted(b, session)
    time.sleep(2)
    assert accepted(a, session)  # inside the renewal threshold: A renews it

    time.sleep(2.5)  # past the first expiry, inside the renewed one

    assert accepted(b, session)
    assert accepted(a, session)


def test_an_idle_session_expires_on_every_node(nodes):
    a, b = nodes
    session = sign_up(a)

    time.sleep(ROLLING_S + 1.5)

    assert not accepted(b, session)
    assert not accepted(a, session)


def test_a_renewal_after_signing_out_elsewhere_renews_nothing():
    from dsp_decikit.auth import sessions

    async def renew_after_sign_out():
        token, info = await sessions.create_session(uuid.uuid4())
        await sessions.delete_session(token)
        return await sessions.renew_session(token, info), await sessions.get_session(token)

    assert asyncio.run(renew_after_sign_out()) == (False, None)


def start_oidc_login(node: str) -> tuple[str, str]:
    """GET the login route; returns (login cookie, state) as the browser would carry them to the provider."""
    response = httpx.get(node + "/api/auth/oidc/login/fake")
    assert response.status_code == 307, response.text
    state = parse_qs(urlparse(response.headers["location"]).query)["state"][0]
    return response.cookies["oidc_session_id"], state


def finish_oidc_login(node: str, login: str, state: str, code: str) -> httpx.Response:
    return httpx.get(node + "/api/auth/oidc/token", params={"code": code, "state": state},
                     cookies={"oidc_session_id": login})


def test_an_oidc_login_started_on_one_node_completes_on_the_other(nodes):
    a, b = nodes
    login, state = start_oidc_login(a)

    response = finish_oidc_login(b, login, state, code=uuid.uuid4().hex[:12])

    assert response.status_code == 307, response.text
    assert accepted(a, response.cookies["session_id"])


def test_an_oidc_login_state_is_used_once_across_nodes(nodes):
    a, b = nodes
    login, state = start_oidc_login(a)
    assert finish_oidc_login(b, login, state, code=uuid.uuid4().hex[:12]).status_code == 307

    replay = finish_oidc_login(a, login, state, code=uuid.uuid4().hex[:12])

    assert (replay.status_code, replay.headers["location"]) == (302, "/unauthorized?detail=Invalid+session")
