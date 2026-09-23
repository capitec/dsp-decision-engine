"""Shared helpers: results.jsonl logging (flushed per line), venv-free import of the plugin."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "decider_trees"))          # the mixed-layout package with its .so
RESULTS = HERE / "results.jsonl"


def log(item: str, **fields):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "item": item, **fields}
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n"); f.flush(); os.fsync(f.fileno())
    return rec
