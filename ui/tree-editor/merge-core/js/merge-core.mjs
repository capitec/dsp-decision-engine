// Live-sync ops from the Rust core compiled to WebAssembly; the arguments cross as JSON text.
// In node call `init({ module: bytes })` once; in a browser `init()` fetches the .wasm next to this file.
import init, { initSync, call } from '../pkg/merge_core.js'

export async function ready(wasm) {
  if (wasm) initSync({ module: wasm })
  else await init()
}

const run = (name, ...args) => JSON.parse(call(name, JSON.stringify(args)))
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
