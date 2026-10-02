"""Resolve and check out source revisions for an experiment run.

An experiment names a revision symbolically (`"HEAD"`, `"HEAD^"`, a tag); the
run resolves it to the immutable git SHA it actually executed and records both.
A scenario that runs on another revision checks out that SHA into a temporary
worktree, runs its flow there in a subprocess, and is removed when the run ends. Revisions the
installed decider cannot read (outside the authored engine window) are rejected
before anything runs.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from packaging.specifiers import SpecifierSet

from decider.exceptions import DeciderError


def resolve(authored: str) -> str | None:
    """Resolve an authored revision (`"HEAD"`, `"HEAD^"`) to a git SHA, or `None` when unresolvable."""
    try:
        out = subprocess.run(["git", "rev-parse", authored], capture_output=True, text=True).stdout.strip()
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
