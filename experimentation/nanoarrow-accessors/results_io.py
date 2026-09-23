import json, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "results.jsonl")

def record(**rec):
    rec.setdefault("ts", time.time())
    with open(PATH, "a") as f:
        f.write(json.dumps(rec) + "\n"); f.flush()
    return rec
