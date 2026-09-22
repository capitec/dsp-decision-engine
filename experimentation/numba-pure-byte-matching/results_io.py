"""Append-only measurement log (same shape as the sibling strand's). One
JSON line per measurement, flushed as it happens, so a killed run still
leaves its numbers behind."""
import json, time, pathlib
PATH = pathlib.Path(__file__).resolve().parent / "results.jsonl"

def record(section, **kv):
    row = {"section": section, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **kv}
    with PATH.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
    print(json.dumps(row, default=str), flush=True)
    return row
