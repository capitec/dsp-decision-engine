// Approach 4 adapter (JS): the custom Rust crate compiled to WebAssembly (wasm-pack --target web), run in node.
import { readFileSync } from 'node:fs'
import * as core from '../../merge-core/js/merge-core.mjs'

export const ready = core.ready(readFileSync(new URL('../../merge-core/pkg/merge_core_bg.wasm', import.meta.url)))

export const { apply, conflicts, diff, isRelated, reconcile, staleOps, touched, rebaseOrders } = core

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
