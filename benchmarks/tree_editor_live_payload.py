"""Wire size and Python codec time of the tree editor's live-sync messages, as JSON, msgpack and protobuf.

Every real tree gets a seeded synthetic edit session; each edit is diffed into one `ops` message, its `applied`
broadcast, and the tree gets one `snapshot`. `--dump DIR` writes the frames for the Node half (`.scratch/jsbench`).

    uv run --with protobuf --with msgpack --with ui/tree-editor/merge-core python benchmarks/tree_editor_live_payload.py [--edits 200] [--seed 1]
        [--trees id,id] [--dump DIR]
"""
import argparse
import copy
import importlib.util
import json
import random
import statistics
import struct
import sys
import time
import uuid
import zlib
from pathlib import Path

import msgpack
from google.protobuf import descriptor_pb2 as D
from google.protobuf import descriptor_pool, message_factory, struct_pb2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui/tree-editor/e2e"))
import real_trees  # noqa: E402

_spec = importlib.util.spec_from_file_location("live_ops", ROOT / "ui/tree-editor/backend/dsp_decikit/services/live_ops.py")
live_ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(live_ops)

WEIGHTS = {"threshold": 50, "rename": 10, "drag": 15, "add": 5, "delete": 5, "output": 10, "param": 5}


def uid(rng):
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


# --- documents as the editor saves them -------------------------------------------------------------------------

def stamp(node, rng, force=False):
    """The editor puts a uuid on every rule node at load."""
    if isinstance(node, dict):
        if force or not node.get("id"):
            node["id"] = uid(rng)
        for k in ("then", "otherwise"):
            stamp(node.get(k), rng, force)
        for b in node.get("branches", []):
            stamp(b, rng, force)


def editor_document(tree, rng):
    doc = copy.deepcopy(tree.document)
    for r in doc.get("rules", []):
        stamp(r["rule"], rng)
    for i, n in enumerate(doc.get("nodes", [])):
        n.setdefault("position", {"x": float(i * 40), "y": float(i % 7 * 60)})
    return doc


# --- synthetic edits: each returns True when it changed the document ---------------------------------------------

def _numbers(doc):
    found = []

    def walk(n):
        if isinstance(n, dict):
            for k, v in n.items():
                if k in ("threshold", "min", "max", "default") and isinstance(v, (int, float)) and not isinstance(v, bool):
                    found.append((n, k))
                elif k not in ("output", "parameters"):
                    walk(v)
        elif isinstance(n, list):
            for x in n:
                walk(x)

    walk(doc)
    return found


def _bump(v, rng):
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + rng.choice([-2, -1, 1, 2])
    if isinstance(v, float):
        return round(v + rng.uniform(-1, 1) * max(abs(v), 1) * 0.3 + 0.001, 3)
    return f"{v}x" if isinstance(v, str) else v


def edit_threshold(doc, rng):
    if not (found := _numbers(doc)):
        return edit_output(doc, rng)
    n, k = rng.choice(found)
    n[k] = _bump(n[k], rng)
    return True


def edit_rename(doc, rng):
    new = f"renamed-{rng.randrange(10**6)}"
    if doc.get("rules"):
        r = rng.choice(doc["rules"])
        r["meta"] = {**(r.get("meta") or {}), "name": new}
    elif doc.get("nodes"):
        data = rng.choice(doc["nodes"]).setdefault("data", {})
        data["meta"] = {**(data.get("meta") or {}), "name": new}
    else:
        doc["name"] = new
    return True


def edit_drag(doc, rng):
    if doc.get("nodes"):
        rng.choice(doc["nodes"])["position"] = {"x": round(rng.uniform(0, 2000), 1), "y": round(rng.uniform(0, 2000), 1)}
        return True
    rules = doc.get("rules", [])
    if len(rules) < 2:
        return False
    rules.insert(rng.randrange(len(rules)), rules.pop(rng.randrange(len(rules))))
    return True


def edit_add(doc, rng):
    if rules := doc.get("rules"):
        new = copy.deepcopy(rng.choice(rules))
        stamp(new["rule"], rng, force=True)
        new["meta"] = {**(new.get("meta") or {}), "name": f"copy-{rng.randrange(10**6)}"}
        rules.insert(len(rules) if rng.random() < 0.7 else rng.randrange(len(rules)), new)
        return True
    if nodes := doc.get("nodes"):
        new, source = copy.deepcopy(rng.choice(nodes)), rng.choice(nodes)["id"]
        new["id"] = uid(rng)
        new["position"] = {"x": round(rng.uniform(0, 2000), 1), "y": round(rng.uniform(0, 2000), 1)}
        nodes.append(new)
        edge = {"source": source, "target": new["id"], "data": {"sourceIndex": 0}}
        if doc["edges"] and "id" in doc["edges"][0]:
            edge["id"] = f"{source}->{new['id']}"
        doc["edges"].append(edge)
        return True
    return False


def edit_delete(doc, rng):
    if len(rules := doc.get("rules", [])) > 1:
        del rules[rng.randrange(len(rules))]
        return True
    if len(nodes := doc.get("nodes", [])) > 1:
        gone = nodes.pop(rng.randrange(len(nodes)))["id"]
        doc["edges"][:] = [e for e in doc["edges"] if gone not in (e["source"], e["target"])]
        if "subtrees" in doc:
            doc["subtrees"][:] = [s for s in doc["subtrees"] if s.get("id") != gone]
        return True
    return False


def edit_output(doc, rng):
    rows = (doc.get("output") or {}).get("data") or []
    if not rows or not (row := rng.choice(rows)):
        return False
    col = rng.choice(sorted(row))
    row[col] = _bump(row[col], rng)
    return True


def edit_param(doc, rng):
    doc.setdefault("parameters", {})[f"p{rng.randrange(10**6)}"] = {"type": "Float64", "default_value": round(rng.random(), 3)}
    return True


EDITS = {"threshold": edit_threshold, "rename": edit_rename, "drag": edit_drag, "add": edit_add,
         "delete": edit_delete, "output": edit_output, "param": edit_param}


def session(tree, edits, seed):
    """`(snapshot document, [(kind, ops)])`: `edits` messages worth of diffs from one seeded session."""
    rng = random.Random(f"{seed}:{tree.id}")
    doc = editor_document(tree, rng)
    snapshot, messages = copy.deepcopy(doc), []
    while len(messages) < edits:
        kind = rng.choices(list(WEIGHTS), list(WEIGHTS.values()))[0]
        new = copy.deepcopy(doc)
        if not EDITS[kind](new, rng):
            kind, new = "threshold", copy.deepcopy(doc)
            if not edit_threshold(new, rng):
                continue
        ops = live_ops.diff(doc, new)
        assert live_ops.apply(doc, ops) == (new, []), (tree.id, kind)
        if ops:
            messages.append((kind, ops))
        doc = new
    return snapshot, messages, rng


def wire_messages(tree, edits, seed):
    """`[(kind, type, message)]` for one tree: the client `ops`, the server `applied` and one `snapshot`."""
    snapshot, edit_list, rng = session(tree, edits, seed)
    client, out = uid(rng), []
    for version, (kind, ops) in enumerate(edit_list, start=1):
        mid = uid(rng)
        out.append((kind, "ops", {"type": "ops", "id": mid, "base": version, "ops": ops}))
        out.append((kind, "applied", {"type": "applied", "version": version + 1, "client": client, "id": mid,
                                      "ops": ops, "dropped": []}))
    out.append(("snapshot", "snapshot", {"type": "snapshot", "version": 1, "doc": snapshot}))
    return out


# --- codecs -----------------------------------------------------------------------------------------------------

# The .proto the Node half loads; the descriptors below are built by hand to match it (no codegen).
PROTO = """syntax = "proto3";
package live%(v)s;
import "google/protobuf/struct.proto";
message Op { string op = 1; string path = 2; %(value)s = 3; }
message Message {
  string type = 1; string id = 2; int64 base = 3; repeated Op ops = 4; int64 version = 5; string client = 6;
  repeated Op dropped = 7; %(doc)s = 8;
}
"""
VARIANTS = {"V": {"value": "google.protobuf.Value value", "doc": "google.protobuf.Value doc"},
            "B": {"value": "bytes value_json", "doc": "bytes doc_json"}}


def _build(v):
    F = D.FieldDescriptorProto
    f = D.FileDescriptorProto(name=f"live{v}.proto", package=f"live{v}", syntax="proto3",
                              dependency=["google/protobuf/struct.proto"])

    def field(m, name, num, typ, type_name=None, repeated=False):
        d = m.field.add(name=name, number=num, type=typ, label=F.LABEL_REPEATED if repeated else F.LABEL_OPTIONAL)
        if type_name:
            d.type_name = type_name

    op, msg = f.message_type.add(name="Op"), f.message_type.add(name="Message")
    field(op, "op", 1, F.TYPE_STRING)
    field(op, "path", 2, F.TYPE_STRING)
    for m, name, num in ((op, "value", 3), (msg, "doc", 8)):
        if v == "V":
            field(m, name, num, F.TYPE_MESSAGE, ".google.protobuf.Value")
        else:
            field(m, name + "_json", num, F.TYPE_BYTES)
    for name, num, typ in (("type", 1, F.TYPE_STRING), ("id", 2, F.TYPE_STRING), ("base", 3, F.TYPE_INT64)):
        field(msg, name, num, typ)
    field(msg, "ops", 4, F.TYPE_MESSAGE, f".live{v}.Op", repeated=True)
    field(msg, "version", 5, F.TYPE_INT64)
    field(msg, "client", 6, F.TYPE_STRING)
    field(msg, "dropped", 7, F.TYPE_MESSAGE, f".live{v}.Op", repeated=True)
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(struct_pb2.DESCRIPTOR.serialized_pb)
    pool.AddSerializedFile(f.SerializeToString())
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(f"live{v}.Message"))


def _to_value(x, val):
    if x is None:
        val.null_value = 0
    elif isinstance(x, bool):
        val.bool_value = x
    elif isinstance(x, (int, float)):
        val.number_value = x
    elif isinstance(x, str):
        val.string_value = x
    elif isinstance(x, list):
        val.list_value.SetInParent()
        for item in x:
            _to_value(item, val.list_value.values.add())
    else:
        val.struct_value.SetInParent()
        for k, item in x.items():
            _to_value(item, val.struct_value.fields[k])


def _from_value(val):
    kind = val.WhichOneof("kind")
    if kind == "number_value":
        n = val.number_value
        return int(n) if n.is_integer() else n
    if kind == "list_value":
        return [_from_value(v) for v in val.list_value.values]
    if kind == "struct_value":
        return {k: _from_value(v) for k, v in val.struct_value.fields.items()}
    return None if kind in (None, "null_value") else getattr(val, kind)


class Proto:
    def __init__(self, v):
        self.cls, self.v = _build(v), v

    def _set(self, holder, name, x):
        if self.v == "V":
            _to_value(x, getattr(holder, name))
        else:
            setattr(holder, name + "_json", _json.encode(x).encode())

    def _get(self, holder, name):
        return _from_value(getattr(holder, name)) if self.v == "V" else json.loads(getattr(holder, name + "_json"))

    def _ops(self, target, ops):
        for o in ops:
            p = target.add(op=o["op"], path=o["path"])
            if "value" in o:
                self._set(p, "value", o["value"])

    def _read_ops(self, ops):
        out = []
        for p in ops:
            o = {"op": p.op, "path": p.path}
            if self.v == "B" and p.value_json or self.v == "V" and p.HasField("value"):
                o["value"] = self._get(p, "value")
            out.append(o)
        return out

    def encode(self, m):
        p = self.cls(type=m["type"])
        if m["type"] == "snapshot":
            p.version = m["version"]
            self._set(p, "doc", m["doc"])
            return p.SerializeToString()
        p.id = m["id"]
        self._ops(p.ops, m["ops"])
        if m["type"] == "ops":
            p.base = m["base"]
        else:
            p.version, p.client = m["version"], m["client"]
            self._ops(p.dropped, m["dropped"])
        return p.SerializeToString()

    def decode(self, b):
        p = self.cls.FromString(b)
        if p.type == "snapshot":
            return {"type": "snapshot", "version": p.version, "doc": self._get(p, "doc")}
        m = {"type": p.type, "id": p.id}
        if p.type == "ops":
            return {**m, "base": p.base, "ops": self._read_ops(p.ops)}
        return {**m, "version": p.version, "client": p.client, "ops": self._read_ops(p.ops),
                "dropped": self._read_ops(p.dropped)}


_json = json.JSONEncoder(separators=(",", ":"))  # what json.dumps(m, separators=...) builds per call


def codecs():
    pv, pb = Proto("V"), Proto("B")
    return {
        "json": (lambda m: _json.encode(m).encode(), json.loads),
        "msgpack": (msgpack.packb, lambda b: msgpack.unpackb(b, raw=False)),
        "protobuf Value": (pv.encode, pv.decode),
        "protobuf json-bytes": (pb.encode, pb.decode),
    }


# --- measuring --------------------------------------------------------------------------------------------------

def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def deflate(b):
    c = zlib.compressobj(6, zlib.DEFLATED, -15)
    return len(c.compress(b) + c.flush())


def best_ns(fn, arg, reps):
    best = None
    for _ in range(reps):
        t = time.perf_counter_ns()
        fn(arg)
        best = min(best or 1 << 62, time.perf_counter_ns() - t)
    return best


def triple(xs, unit=1):
    return f"{pct(xs, 50) / unit:,.0f} / {pct(xs, 95) / unit:,.0f} / {max(xs) / unit:,.0f}"


def measure(trees, edits, seed):
    cs = codecs()
    data = {t.id: wire_messages(t, edits, seed) for t in trees}
    # sizes[tree][type][codec] = [(raw, deflated)]; times[type][codec] = [(enc_ns, dec_ns)]
    sizes = {t: {} for t in data}
    times, drift, ctx = {}, {c: 0 for c in cs}, {c: [] for c in cs}
    for tid, msgs in data.items():
        streams = {c: zlib.compressobj(6, zlib.DEFLATED, -15) for c in cs}
        for _, mtype, m in msgs:
            ref = json.dumps(m, sort_keys=True)
            for name, (enc, dec) in cs.items():
                b = enc(m)
                if json.dumps(dec(b), sort_keys=True) != ref:
                    drift[name] += 1
                sizes[tid].setdefault(mtype, {}).setdefault(name, []).append((len(b), deflate(b)))
                if mtype == "ops":
                    ctx[name].append(len(streams[name].compress(b) + streams[name].flush(zlib.Z_SYNC_FLUSH)))
                reps = 2 if mtype == "snapshot" else 5
                times.setdefault(mtype, {}).setdefault(name, []).append((best_ns(enc, m, reps), best_ns(dec, b, reps)))
    return cs, data, sizes, times, drift, ctx


def report(cs, data, sizes, times, drift, ctx, edits, seed):
    out = [f"{len(data)} trees x {edits} edits, seed {seed}", ""]
    all_msgs = [(k, t, m) for msgs in data.values() for k, t, m in msgs]
    ops_msgs = [m for _, t, m in all_msgs if t == "ops"]
    out.append(f"ops per `ops` message p50 / p95 / max: {triple([len(m['ops']) for m in ops_msgs])}; "
               + ", ".join(f"{k} {sum(1 for kk, t, _ in all_msgs if kk == k and t == 'ops')}" for k in WEIGHTS))
    sized = [(len(json.dumps(m, separators=(",", ":"))), any(o["op"] == "order" for o in m["ops"])) for m in ops_msgs]
    plain, ordered = [n for n, o in sized if not o], [n for n, o in sized if o]
    out += ["", f"`ops` messages with an `order` op (it lists every key of the list): {len(ordered)} of {len(sized)}, "
            f"JSON bytes p50 / p95 / max {triple(ordered)}; the other messages {triple(plain)}"]
    out += ["", "Messages whose decode differs from the original once dumped as JSON (type drift, 5.0 vs 5): "
            + ", ".join(f"{c} {n}" for c, n in drift.items()), ""]
    for mtype, title in (("ops", "client `ops` message"), ("applied", "server `applied` broadcast")):
        out += [f"### Size in bytes, {title}, all trees pooled (p50 / p95 / max)", "",
                "| codec | raw | deflate per message | deflate, context kept |", "|---|---|---|---|"]
        for c in cs:
            pairs = [p for t in sizes.values() for p in t[mtype][c]]
            keep = triple(ctx[c]) if mtype == "ops" else "-"
            out.append(f"| {c} | {triple([r for r, _ in pairs])} | {triple([d for _, d in pairs])} | {keep} |")
        out.append("")
    out += ["### Size in bytes, `ops` message per tree (p50 / p95 / max; raw | deflate per message)", "",
            "| tree | doc bytes | " + " | ".join(cs) + " |", "|---|---|" + "---|" * len(cs)]
    for t, by_type in sizes.items():
        doc = len(json.dumps(data[t][-1][2]["doc"], separators=(",", ":")))
        cells = [f"{triple([r for r, _ in by_type['ops'][c]])} \\| {triple([d for _, d in by_type['ops'][c]])}" for c in cs]
        out.append(f"| {t} | {doc:,} | " + " | ".join(cells) + " |")
    out += ["", "### Snapshot size in bytes (one per tree; p50 / p95 / max over trees)", "",
            "| codec | raw | deflate |", "|---|---|---|"]
    for c in cs:
        pairs = [by_type["snapshot"][c][0] for by_type in sizes.values()]
        out.append(f"| {c} | {triple([r for r, _ in pairs])} | {triple([d for _, d in pairs])} |")
    out += ["", "### Python encode / decode time in microseconds per message (p50 / p95 / max, best of 5)", "",
            "| message | codec | encode | decode | encode+decode p50 |", "|---|---|---|---|---|"]
    for mtype in ("ops", "applied", "snapshot"):
        for c in cs:
            e, d = [x for x, _ in times[mtype][c]], [y for _, y in times[mtype][c]]
            both = [x + y for x, y in times[mtype][c]]
            out.append(f"| {mtype} | {c} | {triple(e, 1000)} | {triple(d, 1000)} | {statistics.median(both) / 1000:,.1f} |")
    return "\n".join(out)


def dump(cs, data, folder):
    """Frames `[u32 length][bytes]` per codec, in one order, plus the index the Node half reads."""
    folder.mkdir(parents=True, exist_ok=True)
    msgs = [(t, mt, m) for tid, ms in data.items() for _, mt, m in ms for t in [tid]]
    (folder / "index.json").write_text(json.dumps([{"tree": t, "type": mt} for t, mt, _ in msgs]))
    for c, (enc, _) in cs.items():
        with open(folder / f"{c.replace(' ', '-')}.bin", "wb") as f:
            for _, _, m in msgs:
                b = enc(m)
                f.write(struct.pack("<I", len(b)) + b)
    for v, parts in VARIANTS.items():
        (folder / f"live{v}.proto").write_text(PROTO % {"v": v, **parts})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--trees")
    ap.add_argument("--dump", type=Path)
    a = ap.parse_args()
    keep = a.trees.split(",") if a.trees else None
    trees = [t for t in real_trees.TREES if keep is None or t.id in keep]
    cs, data, sizes, times, drift, ctx = measure(trees, a.edits, a.seed)
    print(report(cs, data, sizes, times, drift, ctx, a.edits, a.seed))
    if a.dump:
        dump(cs, data, a.dump)


if __name__ == "__main__":
    main()
