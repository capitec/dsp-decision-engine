"""Run every vector in vectors/*.json against a Python adapter and print the failures and a per-semantics summary.

    python run_vectors.py adapters/pure_py.py [--json results.json]

Exits 0 whatever fails. Values compare as JSON: 1 equals 1.0, true is not 1, object key order is ignored except
where a case lists the `keys` its result must have in order.
"""

import argparse
import copy
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(map(same, a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def load_cases():
    cases = []
    for path in sorted((HERE / "vectors").glob("*.json")):
        data = json.loads(path.read_text())
        docs = data.get("docs", {})
        cases += [_resolve(c, docs) for c in data["cases"]]
    return cases


def _resolve(value, docs):
    # A {"$doc": id} value is the shared document `docs[id]`, so a document many cases use is stored once.
    if isinstance(value, dict):
        if value.keys() == {"$doc"}:
            return docs[value["$doc"]]
        return {k: _resolve(v, docs) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, docs) for v in value]
    return value


class Failed(Exception):
    pass


def check(ok, reason):
    if not ok:
        raise Failed(reason)


def unchanged(inputs, before):
    check(same(inputs, before) and json.dumps(inputs) == json.dumps(before), "an input was changed")


def run_case(m, c):
    k = c["kind"]
    inputs = {key: c[key] for key in c if key not in ("name", "kind", "semantics", "expected", "keys")}
    before = copy.deepcopy(inputs)
    if k == "diff":
        ops = m.diff(c["old"], c["new"])
        check(same(ops, c["ops"]), f"diff gave {_short(ops)}")
        doc, dropped = m.apply(c["old"], ops)
        check(same(doc, c["new"]) and dropped == [], f"apply of the diff gave {_short(doc)}, dropped {_short(dropped)}")
    elif k == "apply":
        doc, dropped = m.apply(c["doc"], c["ops"])
        check(same(doc, c["expected"]), f"gave {_short(doc)}")
        check(same(dropped, c["dropped"]), f"dropped {_short(dropped)}")
        check(all(any(d is o for o in c["ops"]) for d in dropped), "dropped ops are not the caller's op objects")
        if "keys" in c:
            check(list(doc) == c["keys"], f"keys in the order {list(doc)}")
        _mutate(doc)
    elif k == "stale":
        stale = m.stale_ops(c["mine"], c["concurrent"])
        check(same(stale, c["dropped"]), f"gave {_short(stale)}")
        check(all(any(d is o for o in c["mine"]) for d in stale), "stale ops are not the caller's op objects")
    elif k == "related":
        check(m.is_related(c["a"], c["b"]) is c["related"] and m.is_related(c["b"], c["a"]) is c["related"],
              f"expected {c['related']}")
    elif k == "conflicts":
        found = m.conflicts(c["mine"], c["theirs"], c["base"])
        check(same(found, c["conflicts"]), f"gave {_short(found)}")
    elif k == "rebase":
        got = m.rebase_orders(c["mine"], c["base"], c["current"])
        check(same(got, c["expected"]), f"gave {_short(got)}")
    elif k == "reconcile":
        got = m.reconcile(c["current"], c["given"])
        check(same(got, c["expected"]), f"gave {_short(got)}")
    elif k == "touched":
        got = m.touched(c["ops"])
        check(same(got, c["expected"]), f"gave {_short(got)}")
    elif k == "raises":
        try:
            got = getattr(m, c["fn"])(*c["args"])
        except NotImplementedError:
            raise
        except Exception as e:
            check(type(e).__name__ == c["error"], f"raised {type(e).__name__}, not {c['error']}")
        else:
            raise Failed(f"returned {_short(got)} instead of raising {c['error']}")
    elif k == "roundtrip":
        doc = m.roundtrip(c["old"], c["new"])
        check(same(doc, c["new"]), f"gave {_short(doc)}")
        _mutate(doc)
    elif k == "online":
        doc, dropped = m.online(c["base"], c["first"], c["second"])
        check(same(doc, c["expected"]["doc"]), f"gave {_short(doc)}")
        check(sorted(dropped) == sorted(c["expected"]["dropped"]), f"dropped {_short(dropped)}")
        _mutate(doc)
    elif k == "offline":
        got, want = m.offline(c["base"], c["mine"], c["theirs"]), c["expected"]
        if "conflicts" in want:
            check(sorted(got.get("conflicts") or []) == sorted(want["conflicts"]), f"gave {_short(got)}")
        else:
            check("merged" in got and same(got["merged"], want["merged"]), f"gave {_short(got)}")
            _mutate(got["merged"])
    elif k == "stored":
        doc = m.export(m.load(c["doc"]))
        check(same(doc, c["doc"]), f"gave {_short(doc)}")
    else:
        raise Failed(f"unknown kind {k}")
    unchanged(inputs, before)


class _Adapter:
    # A function the adapter does not define counts as not implemented.
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        try:
            return getattr(self.module, name)
        except AttributeError:
            raise NotImplementedError(f"the adapter has no {name}") from None


def _mutate(value):
    # Any value the result shares with an input shows up as a changed input.
    if isinstance(value, dict):
        for v in list(value.values()):
            _mutate(v)
        value["__mutated__"] = True
    elif isinstance(value, list):
        for v in value:
            _mutate(v)
        value.append("__mutated__")


def _short(value):
    text = json.dumps(value, default=repr)
    return text if len(text) <= 160 else text[:157] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter")
    parser.add_argument("--json")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("adapter", args.adapter)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(args.adapter).resolve().parent))
    spec.loader.exec_module(m)
    xfail = getattr(m, "XFAIL", {})
    m = _Adapter(m)
    results, counts = [], defaultdict(Counter)
    for c in load_cases():
        try:
            run_case(m, c)
            status, reason = ("xpass", xfail[c["name"]]) if c["name"] in xfail else ("passed", "")
        except NotImplementedError as e:
            status, reason = "not-implemented", str(e)
        except Exception as e:
            status, reason = "failed", f"{type(e).__name__}: {e}" if not isinstance(e, Failed) else str(e)
            if c["name"] in xfail:
                status, reason = "xfail", xfail[c["name"]]
        counts[c["semantics"]][status] += 1
        results.append({"name": c["name"], "kind": c["kind"], "semantics": c["semantics"], "status": status,
                        "reason": reason})
        if status != "passed":
            print(f"{status.upper()} {c['name']} [{c['semantics']}]: {reason}")
    print()
    total = Counter()
    for semantics in sorted(counts):
        total.update(counts[semantics])
        print(f"{semantics:28} " + "  ".join(f"{s} {n}" for s, n in sorted(counts[semantics].items())))
    print(f"{'TOTAL':28} " + "  ".join(f"{s} {n}" for s, n in sorted(total.items())))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
