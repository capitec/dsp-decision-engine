// Run every vector in vectors/*.json against a JS adapter and print the failures and a per-semantics summary.
//
//   node run_vectors.mjs adapters/pure_ts.mjs [--json results.json]
//
// Exits 0 whatever fails. Values compare as JSON (JavaScript has one number type, so 1 and 1.0 are one value),
// object key order is ignored except where a case lists the `keys` its result must have in order.
import { readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const here = new URL('.', import.meta.url)
const args = process.argv.slice(2)
const jsonOut = args.includes('--json') ? args[args.indexOf('--json') + 1] : null
const module = await import(pathToFileURL(resolve(args[0])).href)
const xfail = module.XFAIL ?? {}

class Failed extends Error {}
class NotImplemented extends Error {}

const camel = (name) => name.replace(/_(\w)/g, (_, c) => c.toUpperCase())
function fn(name) {
  const f = module[camel(name)]
  if (typeof f !== 'function') throw new NotImplemented(`the adapter has no ${camel(name)}`)
  return (...a) => {
    try {
      return f(...a)
    } catch (e) {
      if (String(e?.message).startsWith('not implemented')) throw new NotImplemented(e.message)
      throw e
    }
  }
}

function same(a, b) {
  if (typeof a !== typeof b || (a === null) !== (b === null)) return false
  if (Array.isArray(a)) return Array.isArray(b) && a.length === b.length && a.every((x, i) => same(x, b[i]))
  if (a !== null && typeof a === 'object') {
    if (Array.isArray(b)) return false
    const ka = Object.keys(a)
    return ka.length === Object.keys(b).length && ka.every((k) => Object.hasOwn(b, k) && same(a[k], b[k]))
  }
  return a === b
}

// A {"$doc": id} value is the shared document `docs[id]`, so a document many cases use is stored once.
function resolveDocs(value, docs) {
  if (Array.isArray(value)) return value.map((v) => resolveDocs(v, docs))
  if (value === null || typeof value !== 'object') return value
  const keys = Object.keys(value)
  if (keys.length === 1 && keys[0] === '$doc') return docs[value.$doc]
  // fromEntries defines keys, so a `__proto__` key stays data.
  return Object.fromEntries(keys.map((k) => [k, resolveDocs(value[k], docs)]))
}

const short = (v) => {
  const text = JSON.stringify(v) ?? String(v)
  return text.length <= 160 ? text : text.slice(0, 157) + '...'
}
function check(ok, reason) {
  if (!ok) throw new Failed(reason)
}
const isOneOf = (items, list) => items.every((d) => list.some((o) => o === d))

// Any value the result shares with an input shows up as a changed input.
function mutate(value) {
  if (Array.isArray(value)) {
    value.forEach(mutate)
    value.push('__mutated__')
  } else if (value !== null && typeof value === 'object') {
    Object.values(value).forEach(mutate)
    value.__mutated__ = true
  }
}

function runCase(c) {
  const inputs = Object.fromEntries(
    Object.keys(c).filter((k) => !['name', 'kind', 'semantics', 'expected', 'keys'].includes(k)).map((k) => [k, c[k]]),
  )
  const before = JSON.stringify(inputs)
  const k = c.kind
  if (k === 'diff') {
    const ops = fn('diff')(c.old, c.new)
    check(same(ops, c.ops), `diff gave ${short(ops)}`)
    const [doc, dropped] = fn('apply')(c.old, ops)
    check(same(doc, c.new) && dropped.length === 0, `apply of the diff gave ${short(doc)}, dropped ${short(dropped)}`)
  } else if (k === 'apply') {
    const [doc, dropped] = fn('apply')(c.doc, c.ops)
    check(same(doc, c.expected), `gave ${short(doc)}`)
    check(same(dropped, c.dropped), `dropped ${short(dropped)}`)
    check(isOneOf(dropped, c.ops), "dropped ops are not the caller's op objects")
    if (c.keys) check(same(Object.keys(doc), c.keys), `keys in the order ${short(Object.keys(doc))}`)
    mutate(doc)
  } else if (k === 'stale') {
    const stale = fn('stale_ops')(c.mine, c.concurrent)
    check(same(stale, c.dropped), `gave ${short(stale)}`)
    check(isOneOf(stale, c.mine), "stale ops are not the caller's op objects")
  } else if (k === 'related') {
    const related = fn('is_related')
    check(related(c.a, c.b) === c.related && related(c.b, c.a) === c.related, `expected ${c.related}`)
  } else if (k === 'conflicts') {
    const found = fn('conflicts')(c.mine, c.theirs, c.base)
    check(same(found, c.conflicts), `gave ${short(found)}`)
  } else if (k === 'rebase') {
    const got = fn('rebase_orders')(c.mine, c.base, c.current)
    check(same(got, c.expected), `gave ${short(got)}`)
  } else if (k === 'reconcile') {
    const got = fn('reconcile')(c.current, c.given)
    check(same(got, c.expected), `gave ${short(got)}`)
  } else if (k === 'touched') {
    const got = fn('touched')(c.ops)
    check(same(got, c.expected), `gave ${short(got)}`)
  } else if (k === 'raises') {
    const f = fn(c.fn)
    let got
    try {
      got = f(...c.args)
    } catch (e) {
      if (e instanceof NotImplemented) throw e
      return check(JSON.stringify(inputs) === before, 'an input was changed')
    }
    throw new Failed(`returned ${short(got)} instead of throwing (${c.error} in Python)`)
  } else if (k === 'roundtrip') {
    const doc = fn('roundtrip')(c.old, c.new)
    check(same(doc, c.new), `gave ${short(doc)}`)
    mutate(doc)
  } else if (k === 'online') {
    const [doc, dropped] = fn('online')(c.base, c.first, c.second)
    check(same(doc, c.expected.doc), `gave ${short(doc)}`)
    check(same([...dropped].sort(), [...c.expected.dropped].sort()), `dropped ${short(dropped)}`)
    mutate(doc)
  } else if (k === 'offline') {
    const got = fn('offline')(c.base, c.mine, c.theirs)
    if (c.expected.conflicts) {
      check(same([...(got.conflicts ?? [])].sort(), [...c.expected.conflicts].sort()), `gave ${short(got)}`)
    } else {
      check(got.merged !== undefined && same(got.merged, c.expected.merged), `gave ${short(got)}`)
      mutate(got.merged)
    }
  } else if (k === 'stored') {
    const doc = fn('export')(fn('load')(c.doc))
    check(same(doc, c.doc), `gave ${short(doc)}`)
  } else {
    throw new Failed(`unknown kind ${k}`)
  }
  check(JSON.stringify(inputs) === before, 'an input was changed')
  check(Object.keys(Object.prototype).length === 0 && Object.getPrototypeOf({}) === Object.prototype,
    'Object.prototype was changed')
}

const cases = []
for (const file of readdirSync(new URL('vectors/', here)).filter((f) => f.endsWith('.json')).sort()) {
  const data = JSON.parse(readFileSync(new URL(`vectors/${file}`, here), 'utf8'))
  for (const c of data.cases) cases.push(resolveDocs(c, data.docs ?? {}))
}

const results = []
const counts = {}
for (const c of cases) {
  let status = 'passed'
  let reason = ''
  try {
    runCase(c)
    if (c.name in xfail) [status, reason] = ['xpass', xfail[c.name]]
  } catch (e) {
    if (e instanceof NotImplemented) [status, reason] = ['not-implemented', e.message]
    else if (c.name in xfail) [status, reason] = ['xfail', xfail[c.name]]
    else [status, reason] = ['failed', e instanceof Failed ? e.message : `${e?.name}: ${e?.message}`]
  }
  const tally = (counts[c.semantics] ??= {})
  tally[status] = (tally[status] ?? 0) + 1
  results.push({ name: c.name, kind: c.kind, semantics: c.semantics, status, reason })
  if (status !== 'passed') console.log(`${status.toUpperCase()} ${c.name} [${c.semantics}]: ${reason}`)
}
console.log()
const total = {}
const line = (tally) => Object.keys(tally).sort().map((s) => `${s} ${tally[s]}`).join('  ')
for (const semantics of Object.keys(counts).sort()) {
  for (const [s, n] of Object.entries(counts[semantics])) total[s] = (total[s] ?? 0) + n
  console.log(`${semantics.padEnd(28)} ${line(counts[semantics])}`)
}
console.log(`${'TOTAL'.padEnd(28)} ${line(total)}`)
if (jsonOut) writeFileSync(jsonOut, JSON.stringify(results, null, 1))
