"""Differential fuzz: the frozen Python reference against the Rust core (`merge_core`), results compared as exact JSON
text (key order, int vs float). Seeded random edit pairs over the real trees (the editors' edits plus traps: numbers,
odd keys, broken or duplicate list keys) and random op lists, through every function. Errors only on malformed ops
(an error class, or Rust raising where Python passed a malformed op through) are counted apart: the server rejects
such ops before they get here.

Run from ui/tree-editor/backend; `--js-safe` keeps to values JavaScript holds, for `fuzz.mjs` to replay in wasm:

    uv run --no-project --with-requirements requirements.txt --with-editable ../../.. --with ../merge-core \
        python ../merge-core-bench/fuzz.py <seconds> <cases.jsonl> [--js-safe]

`SEED=<n>` changes the seed; `DUMP=<case,...>` writes those cases to the jsonl instead of running them.
"""
import copy
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

TE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TE / "e2e"), str(TE / "backend/tests"), str(TE / "backend")]
spec = importlib.util.spec_from_file_location("ref", TE / "merge-core-bench/reference/live_ops.py")
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
import merge_core as rust  # noqa: E402
from real_trees import TREES  # noqa: E402
from test_live_ops import _edit, _stamp_rule_ids  # noqa: E402

SECONDS, OUT = float(sys.argv[1]), sys.argv[2]
JS_SAFE = "--js-safe" in sys.argv
DUMP = [int(x) for x in os.environ.get("DUMP", "").split(",") if x]
TRAPS = [1, 1.0, True, False, 0, -0.0, None, "", "x", "١", "½", 2**53, float(2**53), 1e300, 0.1, 5e-324, [], {},
         [1, 2], {"a": 1}, "a:b"]
if not JS_SAFE:
    TRAPS += [2**53 + 1, 2**64 + 1, float(2**64), -(2**63) - 1, 10**30]
KEYS = ["a", "b", "__proto__", "a/b", "~0", "~1", "١", "½", "#order", "", "01", "id", "rule", "source", "target"]
if not JS_SAFE:
    KEYS += ["0", "1"]
KEYED = ("rules", "nodes", "edges", "subtrees")


def positions(node, out, depth=0):
    if depth > 7:
        return out
    if isinstance(node, dict):
        for k in list(node):
            out.append((node, k)); positions(node[k], out, depth + 1)
    elif isinstance(node, list):
        for i in range(len(node)):
            out.append((node, i)); positions(node[i], out, depth + 1)
    return out


def containers(doc):
    return [doc] + [p[k] for p, k in positions(doc, []) if isinstance(p[k], (dict, list))]


def mutate(doc, rng):
    pos = positions(doc, [])
    kind = rng.randrange(9)
    if kind == 0 and pos:  # set a trap value
        p, k = rng.choice(pos); p[k] = copy.deepcopy(rng.choice(TRAPS))
    elif kind == 1 and pos:  # delete
        p, k = rng.choice(pos); del p[k]
    elif kind == 2:  # add a key
        d = rng.choice([c for c in containers(doc) if isinstance(c, dict)] or [doc])
        if isinstance(d, dict):
            d[rng.choice(KEYS)] = copy.deepcopy(rng.choice(TRAPS))
    elif kind == 3:  # duplicate or insert into a list (duplicate keys in keyed lists)
        ls = [c for c in containers(doc) if isinstance(c, list)]
        if ls:
            l = rng.choice(ls)
            l.insert(rng.randrange(len(l) + 1), copy.deepcopy(rng.choice(l)) if l and rng.random() < 0.7 else rng.choice(TRAPS))
    elif kind == 4:  # swap/move in a keyed list
        ls = [doc[k] for k in KEYED if isinstance(doc.get(k), list) and len(doc[k]) > 1]
        if ls:
            l = rng.choice(ls); l.insert(rng.randrange(len(l)), l.pop(rng.randrange(len(l))))
    elif kind == 5:  # break or collide a key
        ls = [(k, doc[k]) for k in KEYED if isinstance(doc.get(k), list) and doc[k]]
        if ls:
            name, l = rng.choice(ls); item = rng.choice(l)
            if isinstance(item, dict):
                holder = item.get("rule") if name == "rules" and isinstance(item.get("rule"), dict) else item
                holder["id"] = rng.choice([None, 5, "dup", rng.choice(KEYS)]) if rng.random() < 0.8 else holder.pop("id", None)
    elif kind == 6:  # a small keyed list from scratch
        name = rng.choice(KEYED)
        ids = [rng.choice(["a", "b", "c", "a/b", "١", "0"]) for _ in range(rng.randrange(4))]
        doc[name] = [{"rule": {"id": i}} if name == "rules" else {"id": i, "source": "s", "target": i} for i in ids]
    elif kind == 7 and pos:  # change a leaf to an equal-but-different-type value
        p, k = rng.choice(pos); v = p[k]
        if isinstance(v, bool):
            p[k] = int(v)
        elif isinstance(v, int):
            p[k] = float(v) if abs(v) < 2**1023 else v
        elif isinstance(v, float) and v.is_integer() and abs(v) < 1e300:
            p[k] = int(v)
    else:
        if isinstance(doc.get("rules"), list) or doc.get("nodes"):
            try:
                _edit(doc, rng)
            except Exception:
                pass


def edit(doc, rng, adversarial):
    out = copy.deepcopy(doc)
    for _ in range(rng.randint(1, 3)):
        if adversarial or not (isinstance(out.get("rules"), list) or out.get("nodes")):
            mutate(out, rng)
        else:
            _edit(out, rng)
    return out


def rand_ops(doc, others, rng):
    paths = [o["path"] for o in others if isinstance(o, dict)] + ["", "/", "/rules/0", "/rules/-", "/nodes/01",
             "/a/١", "/x/0/y", "/rules/#order", "/edges/s:t"]
    segs = ["0", "1", "01", "-", "١", "a", "rule", "id", "__proto__", "#order", "", "a~1b"]
    ops = []
    for _ in range(rng.randrange(1, 8)):
        r = rng.random()
        if r < 0.05:
            ops.append(rng.choice([5, None, "x", [], {"op": "add"}, {"op": "add", "path": 5, "value": 1},
                                   {"op": "frob", "path": "/a", "value": 1}, {"op": "order", "path": "/rules", "value": "ab"}]))
            continue
        p = rng.choice(paths) if paths else ""
        if rng.random() < 0.5:
            p = p + "/" + rng.choice(segs)
        if rng.random() < 0.2 and "/" in p[1:]:
            p = p.rsplit("/", 1)[0]
        kind = rng.choice(["add", "replace", "remove", "order"])
        if kind == "order":
            keys = rng.sample(["a", "b", "c", "dup", "0", "s:t"] + [o["path"].rsplit("/", 1)[-1] for o in others[:5] if isinstance(o, dict)], 3)
            ops.append({"op": "order", "path": rng.choice(["/rules", "/nodes", "/edges", "/subtrees", p]), "value": keys})
        elif kind == "remove":
            ops.append({"op": "remove", "path": p})
        else:
            ops.append({"op": kind, "path": p, "value": copy.deepcopy(rng.choice(TRAPS))})
    return ops


def _ascii_only(ref_text, rust_text):
    # The deliberate change: conflicts reads only ASCII digits as an index, so Rust may report fewer paths, each
    # extra one holding a digit of another script.
    a, b = json.loads(ref_text), json.loads(rust_text)
    extra = [p for p in a if p not in b]
    return [p for p in a if p in b] == b and all(
        any(x.isdigit() and not x.isascii() for x in p.split("/")) for p in extra)


def run(mod, base, first, second, rops):
    """Every function on one case; results as exact JSON text (op identity as indices)."""
    out = {}
    def rec(name, f):
        try:
            out[name] = json.dumps(f())
        except Exception as e:
            out[name] = "raises " + type(e).__name__
    d1 = mod.diff(base, first); d2 = mod.diff(base, second)
    out["diff1"], out["diff2"] = json.dumps(d1), json.dumps(d2)
    rec("diff12", lambda: mod.diff(first, second))
    def ap(doc, ops):
        r, dropped = mod.apply(doc, ops)
        return [r, [next(i for i, o in enumerate(ops) if o is d) for d in dropped]]
    rec("apply1", lambda: ap(base, d1))
    rec("apply_rand", lambda: ap(base, rops))
    rec("apply_rand_first", lambda: ap(first, rops))
    def stale(mine, theirs):
        s = mod.stale_ops(mine, theirs)
        return [next(i for i, o in enumerate(mine) if o is x) for x in s]
    rec("stale", lambda: stale(d2, d1))
    rec("stale_rand", lambda: stale(rops, d1))
    rec("stale_rand_rev", lambda: stale(d1, rops))
    def online():
        doc = mod.apply(base, d1)[0]
        st = mod.stale_ops(d2, d1)
        fresh = [o for o in d2 if not any(o is s for s in st)]
        return mod.apply(doc, fresh)[0]
    rec("online", online)
    rec("conflicts", lambda: mod.conflicts(d2, d1, base))
    rec("conflicts_rand", lambda: mod.conflicts(rops, d1, base))
    rec("conflicts_rand_rev", lambda: mod.conflicts(d1, rops, base))
    rec("rebase", lambda: mod.rebase_orders(d2, base, first))
    rec("rebase_rand", lambda: mod.rebase_orders(rops, base, first))
    rec("offline_merge", lambda: mod.apply(first, mod.rebase_orders(d2, base, first))[0])
    rec("touched_rand", lambda: mod.touched(rops))
    return out


def main():
    rng = random.Random(int(os.environ.get("SEED", 20261011)) + (7 if JS_SAFE else 0))
    bases = []
    for t in TREES:
        for stamped in (False, True):
            doc = json.loads(json.dumps(t.document))  # as loaded from JSONB: no objects shared within it
            if stamped:
                _stamp_rule_ids(doc, random.Random(t.id))
            bases.append((f"{t.id}{'+ids' if stamped else ''}", doc, len(json.dumps(doc))))
    small = [b for b in bases if b[2] <= 40_000]
    large = [b for b in bases if b[2] > 40_000]
    synth = [("synthetic", {"rules": [{"rule": {"id": k}, "x": i} for i, k in enumerate("abcd")],
                            "nodes": [{"id": "n1"}, {"id": "n2"}], "edges": [{"source": "n1", "target": "n2"}],
                            "o": {"1": 0, "l": [1, 2, 3]}}, 0)]
    counts, mism = Counter(), []
    stop = time.time() + SECONDS
    n = 0
    with open(OUT, "w") as fh:
        while time.time() < stop:
            r = rng.random()
            name, base, _ = rng.choice(large if r < 0.08 else synth if r < 0.3 else small)
            adv = rng.random() < 0.5
            first, second = edit(base, rng, adv), edit(base, rng, adv)
            rops = rand_ops(base, ref.diff(base, first), rng)
            if DUMP:
                n += 1
                if n in DUMP:
                    fh.write(json.dumps({"case": n, "tree": name, "base": base, "first": first, "second": second, "rops": rops}) + "\n")
                if n >= max(DUMP):
                    return
                continue
            a = run(ref, base, first, second, rops)
            b = run(rust, base, first, second, rops)
            n += 1
            for k in a:
                counts[k] += 1
                if a[k] != b[k]:
                    ra, rb = a[k].startswith("raises "), b[k].startswith("raises ")
                    tag = ("ERRCLASS " if ra and rb else "RUST-RAISES " if rb else
                           "ASCII-INDEX " if not ra and k.startswith("conflicts") and _ascii_only(a[k], b[k]) else "MISMATCH ")
                    counts[tag + k] += 1
                    if tag != "MISMATCH ":
                        continue
                    if len(mism) < 200:
                        mism.append({"case": n, "tree": name, "fn": k, "ref": a[k][:400], "rust": b[k][:400],
                                     "base": base if len(json.dumps(base)) < 3000 else name,
                                     "first": first if len(json.dumps(first)) < 3000 else None,
                                     "second": second if len(json.dumps(second)) < 3000 else None, "rops": rops})
            if len(json.dumps(base)) < 200_000 and n % (1 if JS_SAFE else 4) == 0:
                fh.write(json.dumps({"tree": name, "base": base, "first": first, "second": second, "rops": rops,
                                     "ref": a}) + "\n")
    print(f"cases {n} js_safe={JS_SAFE}")
    for k, v in sorted(counts.items()):
        print(k, v)
    Path(OUT + ".mismatch.json").write_text(json.dumps(mism, indent=1))


main()
