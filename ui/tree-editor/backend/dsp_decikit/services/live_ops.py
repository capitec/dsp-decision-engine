"""Live-sync ops on tree documents: diff, lenient apply, and the online (stale) and offline (conflict) checks.

An op is `{"op": "add"|"replace"|"remove", "path": p, "value": v}` or `{"op": "order", "path": p, "value": [keys]}`.
Paths are JSON pointers, except that items of the keyed lists `/rules` (key `rule.id`), `/nodes` (`id`), `/edges`
(`id`, else `source:target`) and `/subtrees` (`id`) are addressed by key. A keyed list with a missing or duplicate
key is replaced whole (an index there could be read as a key). `ui/src/lib/live-ops.ts` is the TypeScript twin; both run `tests/data/live_ops_cases.json`.

    >>> diff({"rules": [{"rule": {"id": "a"}}]}, {"rules": [{"rule": {"id": "a", "x": 1}}]})
    [{'op': 'add', 'path': '/rules/a/rule/x', 'value': 1}]
"""

import copy

ORDER = "#order"


def _rule_key(item):
    rule = item.get("rule")
    return rule.get("id") if isinstance(rule, dict) else None


def _edge_key(item):
    if item.get("id") is not None:
        return item["id"]
    source, target = item.get("source"), item.get("target")
    return f"{source}:{target}" if isinstance(source, str) and isinstance(target, str) else None


KEYED = {"/rules": _rule_key, "/nodes": lambda i: i.get("id"), "/edges": _edge_key, "/subtrees": lambda i: i.get("id")}


class _Dropped(Exception):
    pass


def _escape(segment):
    return segment.replace("~", "~0").replace("/", "~1")


def _path(segments):
    return "".join("/" + _escape(s) for s in segments)


def _split(path):
    if path == "":
        return []
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"not a path: {path!r}")
    return [s.replace("~1", "/").replace("~0", "~") for s in path.split("/")[1:]]


def _keys(segments, value):
    """The item keys of the list `value` at `segments` when it is a keyed list, else None."""
    key = KEYED.get(_path(segments))
    if key is None or not isinstance(value, list):
        return None
    keys = [key(i) if isinstance(i, dict) else None for i in value]
    if all(isinstance(k, str) for k in keys) and len(set(keys)) == len(keys):
        return keys
    return None


def _same(a, b):
    # JSON leaves: 1 == 1.0 as in JavaScript, but True is not 1.
    return a == b and isinstance(a, bool) == isinstance(b, bool)


def diff(old, new):
    """The ops that turn `old` into `new` (none when they are equal; dict key order is ignored)."""
    ops = []
    _diff(old, new, [], ops)
    return ops


def _diff(old, new, at, ops):
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(old.keys() | new.keys()):
            if k not in new:
                ops.append({"op": "remove", "path": _path([*at, k])})
            elif k not in old:
                ops.append({"op": "add", "path": _path([*at, k]), "value": new[k]})
            else:
                _diff(old[k], new[k], [*at, k], ops)
    elif isinstance(old, list) and isinstance(new, list):
        old_keys, new_keys = _keys(at, old), _keys(at, new)
        if old_keys is not None and new_keys is not None:
            _diff_keyed(old, new, old_keys, new_keys, at, ops)
        elif len(old) == len(new):
            changes = []
            for i, (a, b) in enumerate(zip(old, new)):
                _diff(a, b, [*at, str(i)], changes)
            # Under a keyed path an index would be read as a key, so there a change replaces the whole list.
            whole = changes and _path(at) in KEYED
            ops += [{"op": "replace", "path": _path(at), "value": new}] if whole else changes
        else:
            ops.append({"op": "replace", "path": _path(at), "value": new})
    elif not _same(old, new):
        ops.append({"op": "replace", "path": _path(at), "value": new})


def _diff_keyed(old, new, old_keys, new_keys, at, ops):
    old_items, kept = dict(zip(old_keys, old)), set(new_keys)
    ops += [{"op": "remove", "path": _path([*at, k])} for k in old_keys if k not in kept]
    for k, item in zip(new_keys, new):
        if k in old_items:
            _diff(old_items[k], item, [*at, k], ops)
        else:
            ops.append({"op": "add", "path": _path([*at, k]), "value": item})
    # Apply appends new items, so an order op is needed only when that does not already give the new order.
    appended = [k for k in old_keys if k in kept] + [k for k in new_keys if k not in old_items]
    if appended != new_keys:
        ops.append({"op": "order", "path": _path(at), "value": new_keys})


def reconcile(current, given):
    """`given` keys that exist, in that order, then each missing current key after its current predecessor.

    No current key is ever lost: `reconcile(["x", "a", "b"], ["b", "a"]) == ["x", "b", "a"]`.
    """
    exists = set(current)
    result = list(dict.fromkeys(k for k in given if k in exists))
    placed = set(result)
    for i, k in enumerate(current):
        if k not in placed:
            # Every earlier current key is placed by now, so the nearest placed predecessor is current[i - 1].
            result.insert(result.index(current[i - 1]) + 1 if i else 0, k)
            placed.add(k)
    return result


def apply(doc, ops):
    """`(new_doc, dropped_ops)`: `ops` applied in turn to a copy of `doc`; an op that cannot apply is dropped.

    An op is dropped when its parent does not exist (deleted, index out of range, wrong kind) or it is malformed.
    add/replace set (a new keyed item goes at the end), remove deletes if present, order reconciles.
    """
    # ponytail: copies the whole document per call; copy along the op paths if it shows in latency.
    doc, dropped = copy.deepcopy(doc), []
    for op in ops:
        try:
            doc = _apply_one(doc, op)
        except (_Dropped, ValueError):
            dropped.append(op)
    return doc, dropped


def _apply_one(doc, op):
    if not isinstance(op, dict):
        raise _Dropped
    kind, at = op.get("op"), _split(op.get("path"))
    if kind == "order":
        target, given = _walk(doc, at), op.get("value")
        keys = _keys(at, target)
        if keys is None or not isinstance(given, list) or not all(isinstance(k, str) for k in given):
            raise _Dropped
        items = dict(zip(keys, target))
        target[:] = [items[k] for k in reconcile(keys, given)]
        return doc
    if kind not in ("add", "replace", "remove") or (kind != "remove" and "value" not in op):
        raise _Dropped
    if not at:
        if kind == "remove":
            raise _Dropped
        return copy.deepcopy(op["value"])
    parent, last = _walk(doc, at[:-1]), at[-1]
    if isinstance(parent, dict):
        if kind == "remove":
            parent.pop(last, None)
        else:
            parent[last] = copy.deepcopy(op["value"])
    elif isinstance(parent, list):
        keys = _keys(at[:-1], parent)
        i = _index(parent, at[:-1], last)
        if kind == "remove":
            if i is not None:
                del parent[i]
        elif i is not None:
            parent[i] = copy.deepcopy(op["value"])
        elif keys is not None:
            parent.append(copy.deepcopy(op["value"]))
        else:
            raise _Dropped
    else:
        raise _Dropped
    return doc


def _index(lst, at, segment):
    if _path(at) in KEYED:
        keys = _keys(at, lst)
        return keys.index(segment) if keys is not None and segment in keys else None
    if segment.isascii() and segment.isdigit() and (segment == "0" or segment[0] != "0") and int(segment) < len(lst):
        return int(segment)
    return None


def _walk(doc, segments):
    node = doc
    for n, s in enumerate(segments):
        if isinstance(node, dict) and s in node:
            node = node[s]
        elif isinstance(node, list) and (i := _index(node, segments[:n], s)) is not None:
            node = node[i]
        else:
            raise _Dropped
    return node


def _touch(op):
    return op["path"] + "/" + ORDER if op["op"] == "order" else op["path"]


def touched(ops):
    """The paths `ops` touch, once each; an order op touches `<list>/#order`, not the list."""
    return list(dict.fromkeys(_touch(op) for op in ops))


def _prefix(a, b):
    return len(a) <= len(b) and b[: len(a)] == a


def is_related(a, b):
    """Whether path `a` is equal to, an ancestor of or a descendant of `b`, by segment (`/rules/ab` is not
    under `/rules/a`)."""
    sa, sb = _split(a), _split(b)
    return _prefix(sa, sb) or _prefix(sb, sa)


def stale_ops(mine, concurrent):
    """My ops to drop under the online rule: those with a strict ancestor touched by a concurrent op.

    An equal path or a concurrent edit below mine is not stale: mine applies last and wins.
    """
    theirs = [_split(p) for p in touched(concurrent)]
    stale = []
    for op in mine:
        s = _split(_touch(op))
        if any(len(t) < len(s) and _prefix(t, s) for t in theirs):
            stale.append(op)
    return stale


def conflicts(mine, theirs, base):
    """My touched paths that overlap theirs under the offline rule (both diffed from `base`); empty means merge.

    Overlap is an equal, ancestor or descendant path, unless both sides end with the same value at my path (they
    made the same change). Two order ops on the same list conflict only when both sides changed the relative order
    of the keys common to base, mine and theirs, and changed it differently.
    """
    their_paths = [_split(p) for p in touched(theirs)]
    docs = None
    found = []
    for path in touched(mine):
        s = _split(path)
        hits = [t for t in their_paths if _prefix(s, t) or _prefix(t, s)]
        if not hits:
            continue
        docs = docs or [base, apply(base, mine)[0], apply(base, theirs)[0]]
        if s[-1:] == [ORDER]:
            if s in hits and len(hits) == 1 and not _orders_disagree(docs, s[:-1]):
                continue
        elif not diff(_at(docs[1], s), _at(docs[2], s)):
            continue
        found.append(path)
    return found


def _at(doc, segments):
    try:
        return {"v": _walk(doc, segments)}
    except _Dropped:
        return {}


def _orders_disagree(docs, at):
    keys = []
    for doc in docs:
        try:
            keys.append(_keys(at, _walk(doc, at)))
        except _Dropped:
            return True
    if None in keys:
        return True
    common = set(keys[0]) & set(keys[1]) & set(keys[2])
    base, mine, theirs = ([k for k in ks if k in common] for ks in keys)
    return base != mine != theirs != base


def rebase_orders(mine, base, current):
    """`mine` with each order op that kept the relative order of base's keys (it only placed new keys) rewritten
    onto the order in `current`, so merging it does not undo a reorder made since `base`."""
    rebased = []
    for op in mine:
        if op.get("op") == "order":
            at = _split(op["path"])
            try:
                old, now = _keys(at, _walk(base, at)), _keys(at, _walk(current, at))
            except _Dropped:
                old = now = None
            given = op["value"]
            if old is not None and now is not None and _kept_order(old, given):
                op = {**op, "value": reconcile(given, now)}
        rebased.append(op)
    return rebased


def _kept_order(old, new):
    old_set, new_set = set(old), set(new)
    return [k for k in new if k in old_set] == [k for k in old if k in new_set]
