import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
PROJECTS = sorted(p.name for p in EXAMPLES.iterdir() if (p / "README.md").is_file())


def example_path() -> str:
    return os.pathsep.join(str(p) for p in sorted(EXAMPLES.iterdir()) if p.is_dir())


# One pytest process per project: projects ship identically named test modules and conftests.
@pytest.mark.parametrize("project", PROJECTS)
def test_example_tests_pass(project):
    env = {**os.environ, "PYTHONPATH": example_path()}
    run = subprocess.run([sys.executable, "-m", "pytest", str(EXAMPLES / project), "-q", "-p", "no:cacheprovider"],
                         env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-1000:]
