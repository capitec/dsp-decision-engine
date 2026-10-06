"""Resolve and check out source revisions, and compare two of them.

A revision is named symbolically (`"HEAD"`, `"HEAD^"`, a tag) and resolves to the
immutable git SHA it actually ran. Checking one out makes a temporary worktree
whose flow loads in a subprocess, because two revisions share module names and
only a process that never imported the live code loads the checked-out one.
Revisions the installed decider cannot read (outside the authored engine
window) are rejected before anything runs.
"""
from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
from packaging.specifiers import SpecifierSet

from decider.lab.diff._hash import Snapshot
from decider.lab.diff._outputs import OutputDiff, compare_frames
from decider.lab.diff._structure import StructureDiff, compare
from decider.exceptions import DeciderError


def resolve(authored: str) -> str | None:
    """Resolve an authored revision (`"HEAD"`, `"HEAD^"`) to a git SHA, or `None` when unresolvable."""
    try:
        out = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{authored}^{{commit}}"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return None
    return out or None


def compatible(decider_spec: str, installed: str) -> bool:
    """Whether `installed` satisfies the version specifier `decider_spec` (`">=1.0,<2.0"`); empty means any."""
    if not decider_spec.strip() or not installed:
        return True
    return SpecifierSet(decider_spec).contains(installed, prereleases=True)


def materialise(sha: str) -> Path:
    """Check out `sha` into a temporary detached worktree; the caller imports from it and removes it."""
    tmp = Path(tempfile.mkdtemp(prefix="decider-revision-"))
    try:
        subprocess.run(["git", "worktree", "add", "--detach", str(tmp), sha],
                       capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise DeciderError(f"could not check out revision {sha}: {e}") from e
    return tmp


def remove(tree: Path) -> None:
    """Drop a temporary revision worktree and its checkout directory."""
    try:
        subprocess.run(["git", "worktree", "remove", "--force", str(tree)], capture_output=True, text=True)
    except FileNotFoundError:
        pass
    shutil.rmtree(tree, ignore_errors=True)


class Revisions:
    """Two git revisions of one pipeline, each loaded in its own subprocess.

    Example::

        r = diff.revisions("main", "HEAD", entry="credit.pipeline:build")
        r.structure().changed_steps
        r.outputs(frame, key="application_id").rows_changed
    """

    def __init__(self, base: str, head: str, entry: str):
        self.revisions = (base, head)
        self.entry = entry
        self._snapshots: list[Snapshot | None] = [None, None]

    def structure(self) -> StructureDiff:
        """What changed between the revisions' flows, without running either."""
        for i in (0, 1):
            if self._snapshots[i] is None:
                self._snapshots[i], _ = self._load(i, None, None)
        return compare(*self._snapshots)

    def outputs(self, frame: Any, params: Any = None, key: str | None = None,
                rtol: float = 1e-6, atol: float = 1e-9) -> OutputDiff:
        """Run both revisions over `frame` and report which rows and columns change."""
        frame = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
        runs = []
        for i in (0, 1):
            self._snapshots[i], out = self._load(i, frame, params)
            runs.append(out)
        return compare_frames(*runs, key=key, rtol=rtol, atol=atol)

    def _load(self, i: int, frame: pl.DataFrame | None, params: Any) -> tuple[Snapshot, pl.DataFrame | None]:
        sha = resolve(self.revisions[i])
        if sha is None:
            raise DeciderError(f"revision {self.revisions[i]!r} did not resolve")
        tree = materialise(sha)
        scratch = Path(tempfile.mkdtemp(prefix="decider-diff-"))
        try:
            request = {"entry": self.entry, "tree": str(tree), "params": params, "out": str(scratch)}
            if frame is not None:
                frame.write_parquet(scratch / "input.parquet")
                request["input"] = str(scratch / "input.parquet")
            proc = subprocess.run([sys.executable, "-m", "decider.lab.diff._worker"], input=json.dumps(request),
                                  capture_output=True, text=True)
            try:
                reply = json.loads(proc.stdout.strip().splitlines()[-1])
            except (json.JSONDecodeError, IndexError):
                reply = {"ok": False, "error": f"subprocess exited {proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}"}
            if not reply["ok"]:
                raise DeciderError(f"revision {self.revisions[i]!r}: {reply['error']}")
            snap = pickle.loads((scratch / "snapshot.pkl").read_bytes())
            return snap, pl.read_parquet(scratch / "output.parquet") if frame is not None else None
        finally:
            remove(tree)
            shutil.rmtree(scratch, ignore_errors=True)


def revisions(base: str, head: str, entry: str) -> Revisions:
    """Compare two git revisions of the pipeline that `entry` (`"module:factory"`) builds.

    Example::

        diff.revisions("main", "HEAD", entry="credit.pipeline:build").outputs(frame)
    """
    return Revisions(base, head, entry)
