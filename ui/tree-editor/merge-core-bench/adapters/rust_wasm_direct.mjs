// Approach 4 adapter (JS): the custom Rust crate compiled to WebAssembly without bigint, arguments converted with serde-wasm-bindgen.
import { readFileSync } from 'node:fs'
import { callDirect, initSync } from '../../merge-core/pkg-direct/merge_core.js'

initSync({ module: readFileSync(new URL('../../merge-core/pkg-direct/merge_core_bg.wasm', import.meta.url)) })

const run = (name, ...args) => callDirect(name, args)
const pick = (ops, indices) => indices.map((i) => ops[i])
export const diff = (oldDoc, newDoc) => run('diff', oldDoc, newDoc)
export function apply(doc, ops) {
  const [next, dropped] = run('apply', doc, ops)
  return [next, pick(ops, dropped)]
}
export const staleOps = (mine, concurrent) => pick(mine, run('stale_ops', mine, concurrent))
export const conflicts = (mine, theirs, base) => run('conflicts', mine, theirs, base)
export const rebaseOrders = (mine, base, current) => run('rebase_orders', mine, base, current)
export const isRelated = (a, b) => run('is_related', a, b)
export const touched = (ops) => run('touched', ops)
export const reconcile = (current, given) => run('reconcile', current, given)

// Cases where JavaScript itself, not the core, cannot hold the value.
export const XFAIL = {
  'integers past 2**53 that differ by one differ': 'a JavaScript number holds integers exactly only up to 2**53',
  'an added integer-like key goes last': 'JavaScript objects list integer-like keys first, whatever the insertion order',
}

export const roundtrip = (oldDoc, newDoc) => apply(oldDoc, diff(oldDoc, newDoc))[0]

export function online(base, first, second) {
  const theirs = diff(base, first)
  const mine = diff(base, second)
  const stale = staleOps(mine, theirs)
  const [doc, failed] = apply(apply(base, theirs)[0], mine.filter((op) => !stale.includes(op)))
  return [doc, touched([...stale, ...failed]).sort()]
}

export function offline(base, mine, theirs) {
  const mineOps = diff(base, mine)
  const found = conflicts(mineOps, diff(base, theirs), base)
  if (found.length) return { conflicts: found }
  return { merged: apply(theirs, rebaseOrders(mineOps, base, theirs))[0] }
}

export const load = (doc) => doc
const exportDoc = (state) => state
export { exportDoc as export }
