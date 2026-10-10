/**
 * Live-sync ops on tree documents: diff, lenient apply, and the online (stale) and offline (conflict) checks.
 *
 * The twin of the backend's `services/live_ops.py`; both run `backend/tests/data/live_ops_cases.json`.
 * Paths are JSON pointers, except that items of the keyed lists `/rules` (key `rule.id`), `/nodes` (`id`),
 * `/edges` (`id`, else `source:target`) and `/subtrees` (`id`) are addressed by key. A keyed list with a missing
 * or duplicate key is replaced whole (an index there could be read as a key).
 *
 * @example
 * diff({ rules: [{ rule: { id: 'a' } }] }, { rules: [{ rule: { id: 'a', x: 1 } }] })
 * // [{ op: 'add', path: '/rules/a/rule/x', value: 1 }]
 */

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json }
type Dict = { [key: string]: Json }
/** `add` | `replace` | `remove` (no value) | `order` (value: the list's keys in the new order). */
export type Op = { op: string; path: string; value?: Json }

const ORDER = '#order'

const isDict = (v: unknown): v is Dict => typeof v === 'object' && v !== null && !Array.isArray(v)
const has = (o: object, k: string) => Object.prototype.hasOwnProperty.call(o, k)
const get = (o: Dict, k: string): Json | undefined => (has(o, k) ? o[k] : undefined)
const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v))
// Plain assignment of `__proto__` would set the prototype instead of storing a key.
const setKey = (o: Dict, k: string, v: Json) =>
  Object.defineProperty(o, k, { value: v, writable: true, enumerable: true, configurable: true })

const ruleKey = (i: Dict) => {
  const rule = get(i, 'rule')
  return isDict(rule) ? get(rule, 'id') : undefined
}
const edgeKey = (i: Dict) => {
  const id = get(i, 'id')
  if (id !== undefined && id !== null) return id
  const source = get(i, 'source')
  const target = get(i, 'target')
  return typeof source === 'string' && typeof target === 'string' ? `${source}:${target}` : undefined
}
const idKey = (i: Dict) => get(i, 'id')
const KEYED: Record<string, (item: Dict) => Json | undefined> = {
  '/rules': ruleKey,
  '/nodes': idKey,
  '/edges': edgeKey,
  '/subtrees': idKey,
}

class Dropped extends Error {}

const escape = (s: string) => s.replace(/~/g, '~0').replace(/\//g, '~1')
const toPath = (segments: string[]) => segments.map((s) => '/' + escape(s)).join('')

function split(path: unknown): string[] {
  if (path === '') return []
  if (typeof path !== 'string' || !path.startsWith('/')) throw new Dropped(`not a path: ${String(path)}`)
  return path
    .split('/')
    .slice(1)
    .map((s) => s.replace(/~1/g, '/').replace(/~0/g, '~'))
}

/** The item keys of the list `value` at `segments` when it is a keyed list, else null. */
function keysOf(segments: string[], value: unknown): string[] | null {
  const key = KEYED[toPath(segments)]
  if (!key || !Array.isArray(value)) return null
  const keys = value.map((i) => (isDict(i) ? key(i) : undefined))
  if (keys.every((k): k is string => typeof k === 'string') && new Set(keys).size === keys.length) return keys
  return null
}

/** The ops that turn `oldDoc` into `newDoc` (none when they are equal; object key order is ignored). */
export function diff(oldDoc: Json, newDoc: Json): Op[] {
  const ops: Op[] = []
  walkDiff(oldDoc, newDoc, [], ops)
  return ops
}

function walkDiff(a: Json, b: Json, at: string[], ops: Op[]) {
  if (isDict(a) && isDict(b)) {
    for (const k of [...new Set([...Object.keys(a), ...Object.keys(b)])].sort()) {
      const path = toPath([...at, k])
      if (!has(b, k)) ops.push({ op: 'remove', path })
      else if (!has(a, k)) ops.push({ op: 'add', path, value: b[k] })
      else walkDiff(a[k], b[k], [...at, k], ops)
    }
  } else if (Array.isArray(a) && Array.isArray(b)) {
    const oldKeys = keysOf(at, a)
    const newKeys = keysOf(at, b)
    if (oldKeys && newKeys) diffKeyed(a, b, oldKeys, newKeys, at, ops)
    else if (a.length === b.length) {
      const changes: Op[] = []
      a.forEach((item, i) => walkDiff(item, b[i], [...at, String(i)], changes))
      // Under a keyed path an index would be read as a key, so there a change replaces the whole list.
      if (changes.length && has(KEYED, toPath(at))) ops.push({ op: 'replace', path: toPath(at), value: b })
      else for (const c of changes) ops.push(c)
    } else ops.push({ op: 'replace', path: toPath(at), value: b })
  } else if (a !== b) {
    ops.push({ op: 'replace', path: toPath(at), value: b })
  }
}

function diffKeyed(a: Json[], b: Json[], oldKeys: string[], newKeys: string[], at: string[], ops: Op[]) {
  const oldItems = new Map(oldKeys.map((k, i) => [k, a[i]]))
  const kept = new Set(newKeys)
  for (const k of oldKeys) if (!kept.has(k)) ops.push({ op: 'remove', path: toPath([...at, k]) })
  newKeys.forEach((k, i) => {
    if (oldItems.has(k)) walkDiff(oldItems.get(k)!, b[i], [...at, k], ops)
    else ops.push({ op: 'add', path: toPath([...at, k]), value: b[i] })
  })
  // Apply appends new items, so an order op is needed only when that does not already give the new order.
  const appended = [...oldKeys.filter((k) => kept.has(k)), ...newKeys.filter((k) => !oldItems.has(k))]
  if (appended.some((k, i) => k !== newKeys[i])) ops.push({ op: 'order', path: toPath(at), value: newKeys })
}

/** `given` keys that exist, in that order, then each missing current key after its current predecessor. */
export function reconcile(current: string[], given: string[]): string[] {
  const exists = new Set(current)
  const result = [...new Set(given.filter((k) => exists.has(k)))]
  const placed = new Set(result)
  current.forEach((k, i) => {
    if (placed.has(k)) return
    // Every earlier current key is placed by now, so the nearest placed predecessor is current[i - 1].
    result.splice(i ? result.indexOf(current[i - 1]) + 1 : 0, 0, k)
    placed.add(k)
  })
  return result
}

/**
 * `[newDoc, droppedOps]`: `ops` applied in turn to a copy of `doc`; an op that cannot apply is dropped.
 * An op is dropped when its parent does not exist (deleted, index out of range, wrong kind) or it is malformed.
 */
export function apply(doc: Json, ops: Op[]): [Json, Op[]] {
  // ponytail: copies the whole document per call; copy along the op paths if it shows in latency.
  let result = clone(doc)
  const dropped: Op[] = []
  for (const op of ops) {
    try {
      result = applyOne(result, op)
    } catch (e) {
      if (!(e instanceof Dropped)) throw e
      dropped.push(op)
    }
  }
  return [result, dropped]
}

function applyOne(doc: Json, op: Op): Json {
  if (!isDict(op)) throw new Dropped()
  const at = split(op.path)
  const kind = op.op
  if (kind === 'order') {
    const target = walk(doc, at)
    const keys = keysOf(at, target)
    const given = op.value
    if (!keys || !Array.isArray(given) || !given.every((k) => typeof k === 'string')) throw new Dropped()
    const items = new Map(keys.map((k, i) => [k, (target as Json[])[i]]))
    const order = reconcile(keys, given as string[])
    ;(target as Json[]).splice(0, keys.length, ...order.map((k) => items.get(k)!))
    return doc
  }
  if (!['add', 'replace', 'remove'].includes(kind) || (kind !== 'remove' && !has(op, 'value'))) throw new Dropped()
  if (at.length === 0) {
    if (kind === 'remove') throw new Dropped()
    return clone(op.value as Json)
  }
  const parent = walk(doc, at.slice(0, -1))
  const last = at[at.length - 1]
  if (isDict(parent)) {
    if (kind === 'remove') delete parent[last]
    else setKey(parent, last, clone(op.value as Json))
  } else if (Array.isArray(parent)) {
    const keys = keysOf(at.slice(0, -1), parent)
    const i = indexOf(parent, at.slice(0, -1), last)
    if (kind === 'remove') {
      if (i !== null) parent.splice(i, 1)
    } else if (i !== null) parent[i] = clone(op.value as Json)
    else if (keys) parent.push(clone(op.value as Json))
    else throw new Dropped()
  } else {
    throw new Dropped()
  }
  return doc
}

function indexOf(list: Json[], at: string[], segment: string): number | null {
  if (has(KEYED, toPath(at))) {
    const i = keysOf(at, list)?.indexOf(segment) ?? -1
    return i < 0 ? null : i
  }
  return /^(0|[1-9][0-9]*)$/.test(segment) && Number(segment) < list.length ? Number(segment) : null
}

function walk(doc: Json, segments: string[]): Json {
  let node = doc
  segments.forEach((s, n) => {
    if (isDict(node) && has(node, s)) {
      node = node[s]
      return
    }
    const i = Array.isArray(node) ? indexOf(node, segments.slice(0, n), s) : null
    if (i === null) throw new Dropped()
    node = (node as Json[])[i]
  })
  return node
}

const touch = (op: Op) => (op.op === 'order' ? `${op.path}/${ORDER}` : op.path)

/** The paths `ops` touch, once each; an order op touches `<list>/#order`, not the list. */
export const touched = (ops: Op[]): string[] => [...new Set(ops.map(touch))]

const isPrefix = (a: string[], b: string[]) => a.length <= b.length && a.every((s, i) => s === b[i])

/** Whether path `a` is equal to, an ancestor of or a descendant of `b`, by segment. */
export function isRelated(a: string, b: string): boolean {
  const sa = split(a)
  const sb = split(b)
  return isPrefix(sa, sb) || isPrefix(sb, sa)
}

/** My ops to drop under the online rule: those with a strict ancestor touched by a concurrent op. */
export function staleOps(mine: Op[], concurrent: Op[]): Op[] {
  const theirs = touched(concurrent).map(split)
  return mine.filter((op) => {
    const s = split(touch(op))
    return theirs.some((t) => t.length < s.length && isPrefix(t, s))
  })
}

/**
 * My touched paths that overlap theirs under the offline rule (both diffed from `base`); empty means merge.
 * Two order ops on the same list conflict only when both sides changed the relative order of the keys common to
 * base, mine and theirs, and changed them differently.
 */
export function conflicts(mine: Op[], theirs: Op[], base: Json): string[] {
  const theirPaths = touched(theirs).map(split)
  let docs: Json[] | null = null
  const found: string[] = []
  for (const path of touched(mine)) {
    const s = split(path)
    const hits = theirPaths.filter((t) => isPrefix(s, t) || isPrefix(t, s))
    if (s[s.length - 1] === ORDER && hits.length === 1 && hits[0].length === s.length && isPrefix(s, hits[0])) {
      docs ??= [base, apply(base, mine)[0], apply(base, theirs)[0]]
      if (!ordersDisagree(docs, s.slice(0, -1))) continue
    }
    if (hits.length) found.push(path)
  }
  return found
}

function ordersDisagree(docs: Json[], at: string[]): boolean {
  const keys: string[][] = []
  for (const doc of docs) {
    let k: string[] | null
    try {
      k = keysOf(at, walk(doc, at))
    } catch (e) {
      if (e instanceof Dropped) return true
      throw e
    }
    if (!k) return true
    keys.push(k)
  }
  const common = new Set(keys[0].filter((k) => keys[1].includes(k) && keys[2].includes(k)))
  const [base, mine, theirs] = keys.map((ks) => JSON.stringify(ks.filter((k) => common.has(k))))
  return base !== mine && mine !== theirs && theirs !== base
}
