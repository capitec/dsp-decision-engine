"""Subprocess child: import drv.py BY MODULE NAME (doc 05 SS4.1's fixed
pattern -- sys.path + import, never spec_from_file_location) and call
drv.driver(0.05). Prints one JSON line to stdout.

Run fresh every time (a real, separate interpreter) -- this is what makes it
a clean test of on-disk cache survival, unlike a spec_from_file_location
reload inside one long-lived interpreter.
"""
import json
import os
import sys

workdir = sys.argv[1]
sys.path.insert(0, workdir)

pycache = os.path.join(workdir, "__pycache__")
before_listing = sorted(os.listdir(pycache)) if os.path.isdir(pycache) else []

import drv  # noqa: E402  -- module name import, by design

value = drv.driver(0.05)

after_listing = sorted(os.listdir(pycache)) if os.path.isdir(pycache) else []

print(json.dumps({
    "ok": True,
    "value": value,
    "pycache_before": before_listing,
    "pycache_after": after_listing,
    "co_consts": list(drv.driver.py_func.__code__.co_consts),
}))
