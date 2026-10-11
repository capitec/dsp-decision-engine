"""A one-step decikit project around an exported tree, built and served for real or scored in-process."""

import contextlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import polars as pl
from decikit import ConfigurableStep, flow
from decikit.config import JsonFileStore
from decikit.serving import RequestHandler

from real_trees import REPO

ROOT = REPO / ".scratch/serve-projects"
VERSION = "0.0.1"
PIPELINE = '''from decikit import flow


def build(tree):
    return flow(tree, name="served")
'''


def make(name: str, step: dict, records: list[dict]) -> Path:
    """`.scratch/serve-projects/<name>/`: a package whose `build(tree)` receives `configs/<version>/tree.json`."""
    root = ROOT / name
    shutil.rmtree(root, ignore_errors=True)
    configs = root / "configs" / VERSION
    configs.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "pipeline.py").write_text(PIPELINE)
    (configs / "tree.json").write_text(json.dumps(step, indent=1))
    params = flow(ConfigurableStep.load(step), name="served").parameters().defaults()
    (configs / "params.json").write_text(json.dumps(params))
    (root / "sample_request.json").write_text(json.dumps(records[-1]))
    return root


def records(frame: pl.DataFrame) -> list[dict]:
    """The frame's rows as the JSON objects a client would send (empty ones for a tree that reads nothing)."""
    return [json.loads(json.dumps(row)) for row in frame.drop("probe_row").to_dicts()] or [{}] * frame.height


def in_process(root: Path) -> RequestHandler:
    """The handler `decikit serve` builds, here, interpreted: raw JSON bytes in, the response body out."""
    sys.path.insert(0, str(root.parent))
    handler = RequestHandler(JsonFileStore(basepath=str(root / "configs")), f"{root.name}.pipeline:build",
                             engine="interpreted", sample_request=str(root / "sample_request.json"))
    handler.stage()
    handler.activate()
    return handler


def run(handler: RequestHandler, record: dict) -> dict:
    response = handler.process_fn(json.dumps(record).encode(), "application/json", "application/json")
    return json.loads(response.content)


def _env(root: Path) -> dict:
    return {**os.environ, "DECIKIT_API__CODE_PATH": str(root), "DECIKIT_API__PIPELINE": f"{root.name}.pipeline:build",
            "DECIKIT_CONFIG__BASEPATH": str(root / "configs")}


def build(root: Path) -> None:
    """`decikit build` in the project; fails with its output if it does."""
    done = subprocess.run([sys.executable, "-m", "decikit", "build"], cwd=root, env=_env(root),
                          capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-4000:]


@contextlib.contextmanager
def serving(root: Path, port: int):
    """`decikit serve` on `port` until the block ends; yields a function that POSTs one record."""
    log = (root / "serve.log").open("w")
    server = subprocess.Popen([sys.executable, "-m", "decikit", "serve", "--host", "127.0.0.1", "--port", str(port),
                               "--workers", "1"], cwd=root, env=_env(root), stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_ping(server, url, root)
        yield lambda record: _post(f"{url}/invocations", record)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log.close()


def _wait_for_ping(server: subprocess.Popen, url: str, root: Path) -> None:
    # Staging compiles every kernel before /ping answers, so the wait is as long as the compile.
    for _ in range(600):
        assert server.poll() is None, (root / "serve.log").read_text()[-4000:]
        with contextlib.suppress(OSError):
            with urllib.request.urlopen(f"{url}/ping", timeout=1) as reply:
                if reply.status == 200:
                    return
        with contextlib.suppress(subprocess.TimeoutExpired):
            server.wait(timeout=1)
    raise AssertionError(f"decikit serve never answered /ping:\n{(root / 'serve.log').read_text()[-4000:]}")


def _post(url: str, record: dict) -> dict:
    request = urllib.request.Request(url, json.dumps(record).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as reply:
        return json.loads(reply.read())
