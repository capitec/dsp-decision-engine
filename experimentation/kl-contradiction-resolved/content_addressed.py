"""Step 4: does content-addressed naming close it, given numba's cache is
now shown to be independently responsible (not just CPython's .pyc)?

Same drill as run_2x2's "both_live" arm (worst case: nothing is ever
cleared) except the file is written as drv_<sha256(content)[:16]>.py, and
the module is imported under a name derived from the same hash -- so an
edit that changes content changes both the filename and the registration
name, by construction, with NO cache ever cleared.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen import gen_source  # noqa: E402

PYTHON = sys.executable
CHILD_CA = os.path.join(HERE, "child_ca.py")

with open(CHILD_CA, "w") as f:
    f.write(
        "import json, os, sys\n"
        "workdir, modname = sys.argv[1], sys.argv[2]\n"
        "sys.path.insert(0, workdir)\n"
        "mod = __import__(modname)\n"
        "print(json.dumps({'ok': True, 'value': mod.driver(0.05), "
        "'co_consts': list(mod.driver.py_func.__code__.co_consts)}))\n"
    )

workdir = os.path.join(HERE, "_arm_content_addressed")
shutil.rmtree(workdir, ignore_errors=True)
os.makedirs(workdir)


def write_and_run(threshold):
    src = gen_source(threshold)
    h = hashlib.sha256(src.encode()).hexdigest()[:16]
    modname = f"drv_{h}"
    path = os.path.join(workdir, f"{modname}.py")
    if not os.path.exists(path):  # skip-if-exists, as K finding #4 / doc 05 SS4.2 propose
        with open(path, "w") as f:
            f.write(src)
    proc = subprocess.run([PYTHON, CHILD_CA, workdir, modname], capture_output=True, text=True)
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
    out = json.loads(line)
    out["modname"] = modname
    return out


before = write_and_run("0.0405")
# NOTHING cleared -- worst case for both caches, deliberately.
after_unchanged_redeploy = write_and_run("0.0405")  # same content, "redeploy"
after_edit = write_and_run("0.5950")  # changed content

result = {
    "before": before,
    "unchanged_redeploy_same_hash_path": before["modname"] == after_unchanged_redeploy["modname"],
    "unchanged_redeploy_value": after_unchanged_redeploy["value"],
    "edit_modname_differs": before["modname"] != after_edit["modname"],
    "edit_value_served": after_edit["value"],
    "edit_value_correct": after_edit["value"] == 0.15000000000000002,
    "edit_never_collides_with_before_cache": (
        after_edit["value"] != before["value"] or after_edit["modname"] != before["modname"]
    ),
}
print(json.dumps(result, indent=2))
with open(os.path.join(HERE, "results.jsonl"), "a") as f:
    f.write(json.dumps({"arm": "content_addressed", **result}) + "\n")
