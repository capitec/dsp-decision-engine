"""Append-only measurement log. Every benchmark writes one JSON line per
measurement as it goes, so a killed run still leaves its numbers behind."""
import json, time, pathlib, platform, sys
PATH = pathlib.Path(__file__).resolve().parent / "results.jsonl"

def record(section, **kv):
    row = {"section": section, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **kv}
    with PATH.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    print(json.dumps(row, default=str), flush=True)
    return row
