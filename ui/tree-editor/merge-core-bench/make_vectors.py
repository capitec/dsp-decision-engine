"""Write vectors/*.json and out/bench_inputs.json; every expected result comes from the reference Python copy.

Run from ui/tree-editor/backend (the real trees need decikit):

    uv run --no-project --with-requirements requirements.txt --with-editable ../../.. --with pytest \\
        python ../merge-core-bench/make_vectors.py
"""

import copy
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE_EDITOR = HERE.parent
sys.path[:0] = [str(HERE), str(HERE / "adapters"), str(TREE_EDITOR / "e2e"), str(TREE_EDITOR / "backend/tests"),
               str(TREE_EDITOR / "backend")]

import pure_py as ref  # noqa: E402
import traps  # noqa: E402
from real_trees import BY_ID, TREES  # noqa: E402
from test_live_ops import _edit, _stamp_rule_ids  # noqa: E402

FIXTURE = HERE / "live_ops_cases.json"
# Real trees whose compact JSON is at most this many bytes get seeded edits, so the vector files stay small.
TREE_BYTES = 12_000
STEPS, PAIRS = 3, 2


def _tags(spec):
    return {i: tag for tag, indices in spec.items() for i in indices}


# Case index in live_ops_cases.json -> semantics.
TAGS = _tags({
    "diff-shape": [0, 1, 2, 4, 5, 6, 25], "bool-vs-int": [3],
    "keyed-by-id": [7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 28, 40, 47],
    "keyed-fallback": [13, 14, 15, 16, 17, 18, 23, 39, 41], "path-escaping": [24, 42, 53, 65],
    "drop-inside-deleted": [26, 35, 48], "lww-same-field": [27, 49, 50], "order-reconcile": [29, 30],
    "lenient-apply": [31, 32, 33, 34, 36, 37, 38, 43, 44], "malformed-op": [45, 46],
    "drop-inside-replaced": [51, 54, 56], "path-relation": [52, 57, 58, 59, 60, 61, 62, 63, 64],
    "order-online": [55],
})
# Conflicts case index -> (semantics offline, semantics of the same edits online).
CONFLICT_TAGS = {
    66: ("offline-disjoint-merge", "online-disjoint"), 67: ("offline-overlap-reject", "lww-same-field"),
    68: ("offline-same-value", "lww-same-field"), 69: ("offline-same-value", "drop-inside-replaced"),
    70: ("offline-same-value", "lww-same-field"), 71: ("offline-overlap-reject", "drop-inside-deleted"),
    72: ("offline-overlap-reject", "lww-same-field"), 73: ("path-relation", "path-relation"),
    **{i: ("order-three-way", "order-online") for i in (74, 75, 76, 77, 78, 79, 80, 81, 83)},
    82: ("offline-overlap-reject", "drop-inside-replaced"), 84: ("offline-index-shift", "drop-inside-replaced"),
    85: ("offline-index-shift", "drop-inside-replaced"), 86: ("offline-overlap-reject", "lww-same-field"),
    87: ("offline-disjoint-merge", "online-disjoint"), 88: ("offline-overlap-reject", "lww-same-field"),
}


def online_case(name, semantics, base, first, second):
    doc, dropped = ref.online(base, first, second)
    return {"name": name, "kind": "online", "semantics": semantics, "base": base, "first": first, "second": second,
            "expected": {"doc": doc, "dropped": dropped}}


def offline_case(name, semantics, base, mine, theirs):
    return {"name": name, "kind": "offline", "semantics": semantics, "base": base, "mine": mine, "theirs": theirs,
            "expected": ref.offline(base, mine, theirs)}


def fixture_cases():
    cases = json.loads(FIXTURE.read_text())
    assert len(cases) == 89 and set(TAGS) | set(CONFLICT_TAGS) == set(range(89)), "retag the fixture's cases"
    op_level, neutral = [], []
    for i, c in enumerate(cases):
        tag = TAGS.get(i) or CONFLICT_TAGS[i][0]
        op_level.append({"name": c["name"], "kind": c["kind"], "semantics": tag, **c})
        if c["kind"] == "diff":
            neutral.append({"name": f"roundtrip: {c['name']}", "kind": "roundtrip", "semantics": tag,
                            "old": c["old"], "new": c["new"]})
        elif c["kind"] == "conflicts":
            # The same edits as documents: each side is diffed again from the base.
            mine, theirs = ref.apply(c["base"], c["mine"])[0], ref.apply(c["base"], c["theirs"])[0]
            neutral.append(offline_case(f"offline: {c['name']}", tag, c["base"], mine, theirs))
            neutral.append(online_case(f"online: {c['name']}", CONFLICT_TAGS[i][1], c["base"], theirs, mine))
    return op_level, neutral


class Docs:
    """Documents stored once and referenced as {"$doc": id}."""

    def __init__(self):
        self.docs, self.ids = {}, {}

    def ref(self, doc):
        text = json.dumps(doc)
        if text not in self.ids:
            self.ids[text] = f"d{len(self.ids)}"
            self.docs[self.ids[text]] = json.loads(text)
        return {"$doc": self.ids[text]}

    def share(self, case):
        for key in ("old", "new", "base", "first", "second", "mine", "theirs", "doc"):
            if key in case:
                case[key] = self.ref(case[key])
        expected = case.get("expected", {})
        for key in ("doc", "merged"):
            if key in expected:
                expected[key] = self.ref(expected[key])
        return case


def edited(doc, rng):
    new = copy.deepcopy(doc)
    for _ in range(rng.randint(1, 3)):
        _edit(new, rng)
    return new


def real_tree_cases(docs):
    cases = []
    for tree in TREES:
        size = len(json.dumps(tree.document, separators=(",", ":")))
        print(f"{'seeded' if size <= TREE_BYTES else 'skipped':8} {size:>9} bytes  {tree.id}")
        if size > TREE_BYTES:
            continue
        for stamped in (False, True) if isinstance(tree.document.get("rules"), list) else (False,):
            label = f"{tree.id}{' (rule ids stamped)' if stamped else ''}"
            rng = random.Random(f"vectors-{label}")
            doc = copy.deepcopy(tree.document)
            if stamped:
                _stamp_rule_ids(doc, rng)
            cases.append({"name": f"stored: {label}", "kind": "stored", "semantics": "stored-doc-roundtrip", "doc": doc})
            for step in range(STEPS):
                new = edited(doc, rng)
                cases.append({"name": f"roundtrip: {label} step {step}", "kind": "roundtrip",
                              "semantics": "real-tree-roundtrip", "old": doc, "new": new})
                doc = new
            for pair in range(PAIRS):
                first, second = edited(doc, rng), edited(doc, rng)
                cases.append(online_case(f"online: {label} pair {pair}", "real-tree-online", doc, first, second))
                cases.append(offline_case(f"offline: {label} pair {pair}", "real-tree-offline", doc, first, second))
    return [docs.share(c) for c in cases]


def _rename(doc, items, i, n):
    # One edit of item i of the doc's keyed list: a rule's name, or a node's position.
    item = doc[items][i % n]
    if items == "rules":
        item["meta"] = {**(item.get("meta") or {}), "name": f"bench-{i}"}
    else:
        item["position"] = {"x": float(i), "y": float(i)}


def bench_inputs():
    real = [t for t in TREES if not t.id.startswith("generated-")]
    largest = max(real, key=lambda t: len(json.dumps(t.document)))
    largest_nodes = max((t for t in real if not t.flat), key=lambda t: len(json.dumps(t.document)))
    inputs = {}
    for tree in (BY_ID["generated-521"], largest, largest_nodes):
        base = copy.deepcopy(tree.document)
        _stamp_rule_ids(base, random.Random(tree.id))
        items = "rules" if isinstance(base.get("rules"), list) else "nodes"
        n = len(base[items])
        variants = {"one_edit": [0], "batch50": range(50), "mine50": range(0, 50), "theirs50": range(n // 2, n // 2 + 50),
                    "first": [1], "second": [n - 1]}
        inputs[tree.id] = {"base": base}
        print(f"bench input {tree.id}: {n} {items}")
        for key, edits in variants.items():
            doc = copy.deepcopy(base)
            for i in edits:
                _rename(doc, items, i, n)
            inputs[tree.id][key] = doc
    return inputs


def write(path, data):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n")
    print(f"{path.relative_to(HERE)}: {path.stat().st_size} bytes, {len(data.get('cases', data))} entries")


def main():
    op_level, neutral = fixture_cases()
    write(HERE / "vectors/op_level.json", {"cases": op_level + traps.op_level(ref)})
    write(HERE / "vectors/neutral.json", {"cases": neutral + traps.neutral(ref)})
    docs = Docs()
    cases = real_tree_cases(docs)
    write(HERE / "vectors/real_trees.json", {"docs": docs.docs, "cases": cases})
    write(HERE / "out/bench_inputs.json", bench_inputs())


if __name__ == "__main__":
    main()
