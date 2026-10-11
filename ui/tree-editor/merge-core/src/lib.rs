//! Live-sync ops on tree documents: diff, lenient apply, and the online (stale) and offline (conflict) checks.
//!
//! An op is `{"op": "add"|"replace"|"remove", "path": p, "value": v}` or `{"op": "order", "path": p, "value": [keys]}`.
//! Paths are JSON pointers, except that items of the keyed lists `/rules` (key `rule.id`), `/nodes` (`id`),
//! `/edges` (`id`, else `source:target`) and `/subtrees` (`id`) are addressed by key. A keyed list with a missing or
//! duplicate key is replaced whole (an index there could be read as a key).
//!
//! Ops that a function reports back (dropped, stale) are returned as indices into the slice that was passed in.

use serde_json::{Map, Value};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

mod wire;
#[cfg(feature = "python")]
mod python;
#[cfg(feature = "wasm")]
mod wasm;
pub use wire::{call, call_value};

const ORDER: &str = "#order";

#[derive(Debug, PartialEq)]
pub enum Error {
    Value(String),
    Key(String),
    Type(String),
}

type Res<T> = Result<T, Error>;
struct Dropped;
type Seg = [String];

fn value_err<T>(msg: String) -> Res<T> {
    Err(Error::Value(msg))
}

fn keyed_list(at: &Seg) -> Option<&str> {
    match at {
        [name] if matches!(name.as_str(), "rules" | "nodes" | "edges" | "subtrees") => Some(name),
        _ => None,
    }
}

fn item_key<'a>(list: &str, item: &'a Value) -> Option<Cow<'a, str>> {
    let item = item.as_object()?;
    let text = |v: Option<&'a Value>| v.and_then(Value::as_str).map(Cow::Borrowed);
    match list {
        "rules" => text(item.get("rule")?.as_object()?.get("id")),
        "edges" => match item.get("id") {
            None | Some(Value::Null) => {
                Some(Cow::Owned(format!("{}:{}", item.get("source")?.as_str()?, item.get("target")?.as_str()?)))
            }
            id => text(id),
        },
        _ => text(item.get("id")),
    }
}

/// The item keys of `items` at `at` when it is a keyed list with every key present and unique.
fn valid_keys<'a>(at: &Seg, items: &'a [Value]) -> Option<Vec<Cow<'a, str>>> {
    let list = keyed_list(at)?;
    let keys: Vec<Cow<str>> = items.iter().map(|i| item_key(list, i)).collect::<Option<_>>()?;
    (keys.iter().map(|k| k.as_ref()).collect::<HashSet<_>>().len() == keys.len()).then_some(keys)
}

fn keys(at: &Seg, items: &[Value]) -> Option<Vec<String>> {
    Some(valid_keys(at, items)?.into_iter().map(Cow::into_owned).collect())
}

fn keys_of(at: &Seg, value: &Value) -> Option<Vec<String>> {
    keys(at, value.as_array()?)
}

fn path(at: &Seg) -> String {
    at.iter().map(|s| format!("/{}", s.replace('~', "~0").replace('/', "~1"))).collect()
}

fn split(path: &Value) -> Res<Vec<String>> {
    match path.as_str() {
        Some("") => Ok(vec![]),
        Some(p) if p.starts_with('/') => Ok(p[1..].split('/').map(|s| s.replace("~1", "/").replace("~0", "~")).collect()),
        _ => value_err(format!("not a path: {path}")),
    }
}

fn is_int(n: &serde_json::Number) -> bool {
    !n.to_string().contains(['.', 'e', 'E'])
}

/// Equality of JSON values: `1` equals `1.0` as in JavaScript, `true` does not equal `1`, object key order is ignored.
pub fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) if is_int(x) && is_int(y) => x.to_string() == y.to_string(),
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q)),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w)))
        }
        _ => a == b,
    }
}

fn op(kind: &str, at: &Seg, value: Option<&Value>) -> Value {
    let mut o = Map::new();
    o.insert("op".into(), kind.into());
    o.insert("path".into(), path(at).into());
    if let Some(v) = value {
        o.insert("value".into(), v.clone());
    }
    Value::Object(o)
}

/// The ops that turn `old` into `new` (none when they are equal; object key order is ignored).
pub fn diff(old: &Value, new: &Value) -> Vec<Value> {
    let mut ops = vec![];
    diff_into(old, new, &mut vec![], &mut ops);
    ops
}

fn diff_into(old: &Value, new: &Value, at: &mut Vec<String>, ops: &mut Vec<Value>) {
    match (old, new) {
        (Value::Object(o), Value::Object(n)) => {
            let mut all: Vec<&String> = o.keys().chain(n.keys().filter(|k| !o.contains_key(*k))).collect();
            all.sort();
            for k in all {
                at.push(k.clone());
                match (o.get(k), n.get(k)) {
                    (Some(_), None) => ops.push(op("remove", at, None)),
                    (None, Some(v)) => ops.push(op("add", at, Some(v))),
                    (Some(a), Some(b)) => diff_into(a, b, at, ops),
                    (None, None) => unreachable!(),
                }
                at.pop();
            }
        }
        (Value::Array(o), Value::Array(n)) => {
            if let (Some(ok), Some(nk)) = (keys(at, o), keys(at, n)) {
                diff_keyed(o, n, &ok, &nk, at, ops);
            } else if o.len() == n.len() {
                let mut changes = vec![];
                for (i, (a, b)) in o.iter().zip(n).enumerate() {
                    at.push(i.to_string());
                    diff_into(a, b, at, &mut changes);
                    at.pop();
                }
                // Under a keyed path an index would be read as a key, so there a change replaces the whole list.
                if !changes.is_empty() && keyed_list(at).is_some() {
                    ops.push(op("replace", at, Some(new)));
                } else {
                    ops.extend(changes);
                }
            } else {
                ops.push(op("replace", at, Some(new)));
            }
        }
        _ if same(old, new) => {}
        _ => ops.push(op("replace", at, Some(new))),
    }
}

fn diff_keyed(old: &[Value], new: &[Value], old_keys: &[String], new_keys: &[String], at: &mut Vec<String>, ops: &mut Vec<Value>) {
    let old_items: HashMap<&String, &Value> = old_keys.iter().zip(old).collect();
    let kept: HashSet<&String> = new_keys.iter().collect();
    for k in old_keys.iter().filter(|k| !kept.contains(k)) {
        at.push(k.clone());
        ops.push(op("remove", at, None));
        at.pop();
    }
    for (k, item) in new_keys.iter().zip(new) {
        at.push(k.clone());
        match old_items.get(k) {
            Some(prev) => diff_into(prev, item, at, ops),
            None => ops.push(op("add", at, Some(item))),
        }
        at.pop();
    }
    // Apply appends new items, so an order op is needed only when that does not already give the new order.
    let appended = old_keys.iter().filter(|k| kept.contains(k)).chain(new_keys.iter().filter(|k| !old_items.contains_key(k)));
    if !appended.eq(new_keys.iter()) {
        ops.push(op("order", at, Some(&new_keys.iter().cloned().collect())));
    }
}

/// `given` keys that exist, in that order, then each missing current key after its current predecessor.
///
/// No current key is ever lost: `reconcile(["x", "a", "b"], ["b", "a"]) == ["x", "b", "a"]`.
pub fn reconcile(current: &[String], given: &[String]) -> Vec<String> {
    let exists: HashSet<&String> = current.iter().collect();
    let mut placed = HashSet::new();
    let mut result: Vec<String> = given.iter().filter(|k| exists.contains(k) && placed.insert(*k)).cloned().collect();
    for (i, k) in current.iter().enumerate() {
        if !result.contains(k) {
            // Every earlier current key is placed by now, so the nearest placed predecessor is current[i - 1].
            let at = if i > 0 { result.iter().position(|r| *r == current[i - 1]).map_or(0, |p| p + 1) } else { 0 };
            result.insert(at, k.clone());
        }
    }
    result
}

fn index(items: &[Value], at: &Seg, segment: &str) -> Option<usize> {
    if keyed_list(at).is_some() {
        return valid_keys(at, items)?.iter().position(|k| k == segment);
    }
    let digits = !segment.is_empty() && segment.bytes().all(|b| b.is_ascii_digit());
    let plain = segment == "0" || !segment.starts_with('0');
    segment.parse::<usize>().ok().filter(|&i| digits && plain && i < items.len())
}

fn walk<'a>(doc: &'a Value, at: &Seg) -> Result<&'a Value, Dropped> {
    let mut node = doc;
    for (n, s) in at.iter().enumerate() {
        node = match node {
            Value::Object(m) => m.get(s).ok_or(Dropped)?,
            Value::Array(l) => &l[index(l, &at[..n], s).ok_or(Dropped)?],
            _ => return Err(Dropped),
        };
    }
    Ok(node)
}

fn walk_mut<'a>(doc: &'a mut Value, at: &Seg) -> Result<&'a mut Value, Dropped> {
    let mut node = doc;
    for (n, s) in at.iter().enumerate() {
        node = match node {
            Value::Object(m) => m.get_mut(s).ok_or(Dropped)?,
            Value::Array(l) => {
                let i = index(l, &at[..n], s).ok_or(Dropped)?;
                &mut l[i]
            }
            _ => return Err(Dropped),
        };
    }
    Ok(node)
}

/// A copy of `doc` with `ops` applied in turn, and the indices of the ops that could not apply.
///
/// An op is dropped when its parent does not exist (deleted, index out of range, wrong kind) or it is malformed.
/// add/replace set (a new keyed item goes at the end), remove deletes if present, order reconciles.
pub fn apply(doc: &Value, ops: &[Value]) -> (Value, Vec<usize>) {
    let mut doc = doc.clone();
    let dropped = apply_in_place(&mut doc, ops);
    (doc, dropped)
}

/// `apply` on `doc` itself, for a caller that owns it already.
pub fn apply_in_place(doc: &mut Value, ops: &[Value]) -> Vec<usize> {
    (0..ops.len()).filter(|&i| apply_one(doc, &ops[i]).is_err()).collect()
}

fn apply_one(doc: &mut Value, op: &Value) -> Result<(), Dropped> {
    let op = op.as_object().ok_or(Dropped)?;
    let at = split(op.get("path").unwrap_or(&Value::Null)).map_err(|_| Dropped)?;
    let kind = op.get("op").and_then(Value::as_str).unwrap_or("");
    let value = op.get("value");
    if kind == "order" {
        let Value::Array(items) = walk_mut(doc, &at)? else { return Err(Dropped) };
        let keys = keys(&at, items).ok_or(Dropped)?;
        let given = value.and_then(Value::as_array).ok_or(Dropped)?;
        let given: Vec<String> = given.iter().map(|k| k.as_str().map(String::from)).collect::<Option<_>>().ok_or(Dropped)?;
        let mut by_key: HashMap<String, Value> = keys.iter().cloned().zip(items.drain(..)).collect();
        items.extend(reconcile(&keys, &given).iter().filter_map(|k| by_key.remove(k)));
        return Ok(());
    }
    if !matches!(kind, "add" | "replace" | "remove") || (kind != "remove" && value.is_none()) {
        return Err(Dropped);
    }
    let value = if kind == "remove" { None } else { value };
    let Some((last, parent_at)) = at.split_last() else {
        if kind == "remove" {
            return Err(Dropped);
        }
        *doc = value.cloned().ok_or(Dropped)?;
        return Ok(());
    };
    match walk_mut(doc, parent_at)? {
        Value::Object(m) => match value {
            None => {
                m.shift_remove(last);
            }
            Some(v) => {
                m.insert(last.clone(), v.clone());
            }
        },
        Value::Array(l) => match (index(l, parent_at, last), value) {
            (Some(i), None) => {
                l.remove(i);
            }
            (Some(i), Some(v)) => l[i] = v.clone(),
            (None, None) => {}
            (None, Some(v)) if valid_keys(parent_at, l).is_some() => l.push(v.clone()),
            _ => return Err(Dropped),
        },
        _ => return Err(Dropped),
    }
    Ok(())
}

fn touch(op: &Value) -> Res<String> {
    let Value::Object(o) = op else { return Err(Error::Type(format!("not an op: {op}"))) };
    let kind = o.get("op").ok_or_else(|| Error::Key("op".into()))?;
    let p = o.get("path").ok_or_else(|| Error::Key("path".into()))?;
    // ponytail: a non-string path is a ValueError here (the Python twin returned it as is, then failed in split).
    let p = p.as_str().ok_or_else(|| Error::Value(format!("not a path: {p}")))?;
    Ok(if kind == "order" { format!("{p}/{ORDER}") } else { p.to_string() })
}

/// The paths `ops` touch, once each; an order op touches `<list>/#order`, not the list.
pub fn touched(ops: &[Value]) -> Res<Vec<String>> {
    let mut seen = HashSet::new();
    ops.iter().map(touch).filter(|p| p.as_ref().map_or(true, |p| seen.insert(p.clone()))).collect()
}

fn touched_split(ops: &[Value]) -> Res<Vec<Vec<String>>> {
    touched(ops)?.iter().map(|p| split(&Value::from(p.as_str()))).collect()
}

fn prefix(a: &Seg, b: &Seg) -> bool {
    a.len() <= b.len() && b[..a.len()] == *a
}

/// Whether path `a` is equal to, an ancestor of or a descendant of `b`, by segment (`/rules/ab` is not under `/rules/a`).
pub fn is_related(a: &Value, b: &Value) -> Res<bool> {
    let (sa, sb) = (split(a)?, split(b)?);
    Ok(prefix(&sa, &sb) || prefix(&sb, &sa))
}

/// Indices of my ops to drop under the online rule: those with a strict ancestor touched by a concurrent op.
///
/// An equal path or a concurrent edit below mine is not stale: mine applies last and wins.
pub fn stale_ops(mine: &[Value], concurrent: &[Value]) -> Res<Vec<usize>> {
    let theirs = touched_split(concurrent)?;
    let mut stale = vec![];
    for (i, o) in mine.iter().enumerate() {
        let s = split(&Value::from(touch(o)?))?;
        if theirs.iter().any(|t| t.len() < s.len() && prefix(t, &s)) {
            stale.push(i);
        }
    }
    Ok(stale)
}

/// My touched paths that overlap theirs under the offline rule (both diffed from `base`); empty means merge.
///
/// Overlap is an equal, ancestor or descendant path, unless both sides end with the same value at my path (they made
/// the same change). Two order ops on the same list conflict only when both sides changed the relative order of the
/// keys common to base, mine and theirs, and changed it differently.
pub fn conflicts(mine: &[Value], theirs: &[Value], base: &Value) -> Res<Vec<String>> {
    let their_paths = touched_split(theirs)?;
    let mut docs: Option<[Value; 2]> = None;
    let mut found = vec![];
    for p in touched(mine)? {
        let s = split(&Value::from(p.as_str()))?;
        let hits: Vec<&Vec<String>> = their_paths.iter().filter(|t| prefix(&s, t) || prefix(t, &s)).collect();
        if hits.is_empty() {
            continue;
        }
        let docs = docs.get_or_insert_with(|| [apply(base, mine).0, apply(base, theirs).0]);
        let docs = [base, &docs[0], &docs[1]];
        if s.last().is_some_and(|l| l == ORDER) {
            if hits.len() == 1 && *hits[0] == s && !orders_disagree(&docs, &s[..s.len() - 1]) {
                continue;
            }
        } else {
            let same_end = match (walk(docs[1], &s), walk(docs[2], &s)) {
                (Ok(a), Ok(b)) => same(a, b),
                (Err(_), Err(_)) => true,
                _ => false,
            };
            // Under something they replaced, an index can name another item on their side than on mine.
            // ponytail: is_numeric is a superset of Python's str.isdigit (it also takes 1/2 and Roman numerals): errs to a conflict.
            let indexed = hits.iter().any(|t| t.len() < s.len() && s[t.len()..].iter().any(|x| x.chars().next().is_some() && x.chars().all(char::is_numeric)));
            if same_end && !indexed {
                continue;
            }
        }
        found.push(p);
    }
    Ok(found)
}

fn orders_disagree(docs: &[&Value; 3], at: &Seg) -> bool {
    let mut all = vec![];
    for doc in docs {
        match walk(doc, at).ok().and_then(|l| keys_of(at, l)) {
            Some(k) => all.push(k),
            None => return true,
        }
    }
    let common: HashSet<&String> = all[0].iter().filter(|k| all[1].contains(k) && all[2].contains(k)).collect();
    let [base, mine, theirs]: [Vec<&String>; 3] = std::array::from_fn(|i| all[i].iter().filter(|k| common.contains(k)).collect());
    base != mine && mine != theirs && theirs != base
}

/// `mine` with each order op that kept the relative order of base's keys (it only placed new keys) rewritten onto the
/// order in `current`, so merging it does not undo a reorder made since `base`.
pub fn rebase_orders(mine: &[Value], base: &Value, current: &Value) -> Res<Vec<Value>> {
    let mut rebased = vec![];
    for o in mine {
        let Value::Object(m) = o else { return Err(Error::Type(format!("not an op: {o}"))) };
        let mut o = o.clone();
        if m.get("op").is_some_and(|k| k == "order") {
            let at = split(m.get("path").ok_or_else(|| Error::Key("path".into()))?)?;
            let given = m.get("value").ok_or_else(|| Error::Key("value".into()))?;
            let old = walk(base, &at).ok().and_then(|l| keys_of(&at, l));
            let now = walk(current, &at).ok().and_then(|l| keys_of(&at, l));
            if let (Some(old), Some(now)) = (old, now) {
                let given = strings(given)?;
                if kept_order(&old, &given) {
                    o["value"] = reconcile(&given, &now).into();
                }
            }
        }
        rebased.push(o);
    }
    Ok(rebased)
}

fn strings(v: &Value) -> Res<Vec<String>> {
    let items = v.as_array().ok_or_else(|| Error::Type(format!("not a list: {v}")))?;
    items.iter().map(|k| k.as_str().map(String::from).ok_or_else(|| Error::Type(format!("not a key: {k}")))).collect()
}

fn kept_order(old: &[String], new: &[String]) -> bool {
    let (old_set, new_set): (HashSet<&String>, HashSet<&String>) = (old.iter().collect(), new.iter().collect());
    new.iter().filter(|k| old_set.contains(k)).eq(old.iter().filter(|k| new_set.contains(k)))
}
