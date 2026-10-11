"""Trap cases: JSON numbers, key order, `__proto__`, non-ASCII digits, malformed ops, aliasing, identity, rebase."""

BIG = 2**64 + 1  # past every fixed-width integer
SAFE = 2**53  # the last integer a JavaScript number holds exactly


def _rules(*keys):
    return {"rules": [{"rule": {"id": k}} for k in keys]}


def _diff(ref, name, semantics, old, new):
    return {"name": name, "kind": "diff", "semantics": semantics, "old": old, "new": new, "ops": ref.diff(old, new)}


def _apply(ref, name, semantics, doc, ops, keys=False):
    result, dropped = ref.apply(doc, ops)
    case = {"name": name, "kind": "apply", "semantics": semantics, "doc": doc, "ops": ops, "expected": result,
            "dropped": dropped}
    return {**case, "keys": list(result)} if keys else case


def _conflicts(ref, name, semantics, base, mine, theirs):
    return {"name": name, "kind": "conflicts", "semantics": semantics, "base": base, "mine": mine, "theirs": theirs,
            "conflicts": ref.conflicts(mine, theirs, base)}


def _raises(ref, name, fn, *args):
    try:
        getattr(ref, fn)(*args)
    except Exception as e:
        return {"name": name, "kind": "raises", "semantics": "malformed-op", "fn": fn, "args": list(args),
                "error": type(e).__name__}
    raise AssertionError(f"{name}: the reference does not raise")


def _replace(path, value):
    return {"op": "replace", "path": path, "value": value}


def op_level(ref):
    gone = _replace("/gone/x", 1)
    return [
        _diff(ref, "1 and 1.0 are one value", "numbers-1-vs-1.0", {"a": 1, "b": [2]}, {"a": 1.0, "b": [2.0]}),
        _conflicts(ref, "1 and 1.0 on both sides are the same change", "numbers-1-vs-1.0",
                   {"a": 0}, [_replace("/a", 1)], [_replace("/a", 1.0)]),
        _conflicts(ref, "true and 1 on both sides conflict", "bool-vs-int",
                   {"a": 0}, [_replace("/a", True)], [_replace("/a", 1)]),
        _diff(ref, "integers past 2**53 that differ by one differ", "large-int", {"a": SAFE}, {"a": SAFE + 1}),
        _diff(ref, "an integer past 2**64 is kept", "large-int", {"a": 1}, {"a": BIG}),
        _diff(ref, "an integer past 2**53 and the float it rounds to differ", "large-int", {"a": SAFE + 1},
              {"a": float(SAFE + 1)}),
        _conflicts(ref, "an integer past 2**53 and the float it rounds to on the two sides conflict", "large-int",
                   {"a": 0}, [_replace("/a", SAFE + 1)], [_replace("/a", float(SAFE + 1))]),
        _apply(ref, "apply keeps an integer past 2**64 exactly", "large-int", {}, [{"op": "add", "path": "/a",
                                                                                  "value": BIG}]),
        _apply(ref, "remove keeps the order of the other keys, replace keeps the key's place, add appends",
               "key-order", {"b": 1, "a": 2, "c": 3, "d": 4},
               [{"op": "remove", "path": "/a"}, _replace("/b", 9), {"op": "add", "path": "/e", "value": 5}], keys=True),
        _apply(ref, "an added integer-like key goes last", "key-order", {"b": 1},
               [{"op": "add", "path": "/1", "value": 2}], keys=True),
        _diff(ref, "diff orders keys by code point", "key-order", {}, {"￿": 1, "\U0001f600": 2, "a": 3}),
        _apply(ref, "a __proto__ key is stored as data", "proto-key", {},
               [{"op": "add", "path": "/__proto__", "value": {"polluted": True}}], keys=True),
        _apply(ref, "a path through a __proto__ key resolves to the data", "proto-key", {"__proto__": {"a": 1}},
               [_replace("/__proto__/a", 2)]),
        _diff(ref, "a __proto__ key is diffed as data", "proto-key", {"__proto__": {"a": 1}}, {"__proto__": {"a": 2}}),
        _apply(ref, "a non-ASCII digit is not a list index", "nonascii-digit-index", {"l": [1, 2]},
               [_replace("/l/١", 9), _replace("/l/１", 9), _replace("/l/1", 9)]),
        _conflicts(ref, "an Arabic-Indic digit under their replace counts as an index", "nonascii-digit-index",
                   {"a": {"١": 0}}, [_replace("/a/١", 5)], [_replace("/a", {"١": 5})]),
        _conflicts(ref, "an ASCII digit under their replace counts as an index", "nonascii-digit-index",
                   {"a": {"1": 0}}, [_replace("/a/1", 5)], [_replace("/a", {"1": 5})]),
        _conflicts(ref, "a key that is not a digit under their replace with the same value merges",
                   "nonascii-digit-index", {"a": {"x": 0}}, [_replace("/a/x", 5)], [_replace("/a", {"x": 5})]),
        _apply(ref, "malformed ops are dropped", "malformed-op", {"a": 1},
               [5, None, "x", [], {"op": "add"}, {"op": "add", "path": 5, "value": 1}, {"op": "add", "path": "/b"},
                {"path": "/b", "value": 1}, {"op": "order", "path": "/rules", "value": "ab"}, _replace("/a", 2)]),
        _raises(ref, "is_related of a path without a slash raises", "is_related", "a", "/b"),
        _raises(ref, "is_related of a number raises", "is_related", 5, "/b"),
        _raises(ref, "stale_ops of a path without a slash raises", "stale_ops",
                [{"op": "add", "path": "x", "value": 1}], [_replace("/a", 1)]),
        _raises(ref, "stale_ops of an op without a path raises", "stale_ops", [{"op": "remove"}], [_replace("/a", 1)]),
        _raises(ref, "conflicts of their path without a slash raises", "conflicts",
                [_replace("/a", 1)], [_replace("a", 1)], {}),
        _raises(ref, "touched of an op without a path raises", "touched", [{"op": "add", "value": 1}]),
        _raises(ref, "touched of an op that is not an object raises", "touched", [5]),
        _apply(ref, "the result shares nothing with the document or the ops", "no-aliasing", {"a": {"x": [1]}},
               [{"op": "add", "path": "/b", "value": {"y": [2]}}, {"op": "add", "path": "", "value": {"r": {"z": [3]}}},
                {"op": "add", "path": "/s", "value": {"w": [4]}}]),
        _apply(ref, "dropped ops are the caller's op objects", "dropped-identity", {"a": 1},
               [_replace("/a", 2), gone, _replace("/a", 3), gone]),
        {"name": "stale ops are the caller's op objects", "kind": "stale", "semantics": "dropped-identity",
         "mine": [_replace("/r/x", 1), _replace("/s", 1)], "concurrent": [{"op": "remove", "path": "/r"}],
         "dropped": ref.stale_ops([_replace("/r/x", 1), _replace("/s", 1)], [{"op": "remove", "path": "/r"}])},
        {"name": "touched lists each path once, an order op as <list>/#order", "kind": "touched",
         "semantics": "order-online", "ops": [{"op": "order", "path": "/rules", "value": []}, _replace("/a", 1),
                                              _replace("/a", 2)],
         "expected": ref.touched([{"op": "order", "path": "/rules", "value": []}, _replace("/a", 1)])},
        *(_reconcile(ref, current, given) for current, given in
          [(["x", "a", "b"], ["b", "a"]), (["a", "b", "c"], ["c", "c", "z", "a"]), ([], ["a"]), (["a", "b"], [])]),
        _rebase(ref, "my insert merged onto their reorder keeps their order",
                [{"op": "add", "path": "/rules/d", "value": {"rule": {"id": "d"}}},
                 {"op": "order", "path": "/rules", "value": ["a", "d", "b", "c"]}], _rules("a", "b", "c"),
                _rules("c", "b", "a")),
        _rebase(ref, "my reorder is not rebased", [{"op": "order", "path": "/rules", "value": ["c", "a", "b"]}],
                _rules("a", "b", "c"), _rules("a", "e", "b", "c")),
        _rebase(ref, "an order op on a list that is gone is kept as is",
                [{"op": "order", "path": "/rules", "value": ["a"]}], _rules("a"), {}),
    ]


def _reconcile(ref, current, given):
    return {"name": f"reconcile {current} {given}", "kind": "reconcile", "semantics": "order-reconcile",
            "current": current, "given": given, "expected": ref.reconcile(current, given)}


def _rebase(ref, name, mine, base, current):
    return {"name": name, "kind": "rebase", "semantics": "order-rebase", "mine": mine, "base": base,
            "current": current, "expected": ref.rebase_orders(mine, base, current)}


def neutral(ref):
    def roundtrip(name, semantics, old, new):
        return {"name": f"roundtrip: {name}", "kind": "roundtrip", "semantics": semantics, "old": old, "new": new}

    def offline(name, semantics, base, mine, theirs):
        return {"name": f"offline: {name}", "kind": "offline", "semantics": semantics, "base": base, "mine": mine,
                "theirs": theirs, "expected": ref.offline(base, mine, theirs)}

    def online(name, semantics, base, first, second):
        doc, dropped = ref.online(base, first, second)
        return {"name": f"online: {name}", "kind": "online", "semantics": semantics, "base": base, "first": first,
                "second": second, "expected": {"doc": doc, "dropped": dropped}}

    def stored(name, semantics, doc):
        return {"name": f"stored: {name}", "kind": "stored", "semantics": semantics, "doc": doc}

    odd = {"__proto__": {"polluted": True}, "a/b~c": [1.5, -0.0, None, True, "١"], "￿": {}, "big": BIG}
    return [
        roundtrip("true and 1 swap", "bool-vs-int", {"a": 1, "b": True}, {"a": True, "b": 1}),
        roundtrip("an integer past 2**64", "large-int", {"a": 1}, {"a": BIG}),
        roundtrip("a __proto__ key", "proto-key", {}, {"__proto__": {"polluted": True}}),
        roundtrip("keys with / and ~", "path-escaping", {"a/b~c": 1}, {"a/b~c": 2, "~1": {"/": 3}}),
        stored("odd keys and values", "stored-doc-roundtrip", odd),
        offline("1 and 1.0 on both sides merge", "numbers-1-vs-1.0", {"a": 0}, {"a": 1}, {"a": 1.0}),
        offline("true and 1 on both sides conflict", "bool-vs-int", {"a": 0}, {"a": True}, {"a": 1}),
        offline("my insert onto their reorder keeps their order", "order-rebase", _rules("a", "b", "c"),
                _rules("a", "d", "b", "c"), _rules("c", "b", "a")),
        online("my edit inside a rule they deleted is dropped", "drop-inside-deleted",
               {"rules": [{"rule": {"id": "a", "x": 1}}, {"rule": {"id": "b"}}]}, _rules("b"),
               {"rules": [{"rule": {"id": "a", "x": 2}}, {"rule": {"id": "b"}}]}),
        online("the last writer of a field wins", "lww-same-field", {"a": 0, "b": 0}, {"a": 1, "b": 0},
               {"a": 2, "b": 0}),
    ]
