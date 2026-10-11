/**
 * Live-sync ops on tree documents: diff, lenient apply, and the online (stale) and offline (conflict) checks.
 *
 * The Rust crate `merge-core` compiled to WebAssembly, the same code the backend runs as a Python extension.
 * Call `loadLiveOps()` once before the first use (the app does it before rendering). Documents cross the boundary
 * as JSON text; `apply` and `staleOps` return the caller's own op objects.
 *
 * @example
 * await loadLiveOps()
 * diff({ rules: [{ rule: { id: 'a' } }] }, { rules: [{ rule: { id: 'a', x: 1 } }] })
 * // [{ op: 'add', path: '/rules/a/rule/x', value: 1 }]
 */
import init, { call } from '../../../merge-core/pkg/merge_core.js'
import wasmUrl from '../../../merge-core/pkg/merge_core_bg.wasm?url'

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json }
/** `add` | `replace` | `remove` (no value) | `order` (value: the list's keys in the new order). */
export type Op = { op: string; path: string; value?: Json }

export const loadLiveOps = () => init({ module_or_path: wasmUrl }).then(() => undefined)

const run = <T>(name: string, ...args: unknown[]): T => JSON.parse(call(name, JSON.stringify(args)))
const pick = (ops: Op[], indices: number[]) => indices.map((i) => ops[i])

export const diff = (oldDoc: Json, newDoc: Json): Op[] => run('diff', oldDoc, newDoc)
export const reconcile = (current: string[], given: string[]): string[] => run('reconcile', current, given)
export function apply(doc: Json, ops: Op[]): [Json, Op[]] {
  const [next, dropped] = run<[Json, number[]]>('apply', doc, ops)
  return [next, pick(ops, dropped)]
}
export const touched = (ops: Op[]): string[] => run('touched', ops)
export const isRelated = (a: string, b: string): boolean => run('is_related', a, b)
export const staleOps = (mine: Op[], concurrent: Op[]): Op[] => pick(mine, run<number[]>('stale_ops', mine, concurrent))
export const conflicts = (mine: Op[], theirs: Op[], base: Json): string[] => run('conflicts', mine, theirs, base)
export const rebaseOrders = (mine: Op[], base: Json, current: Json): Op[] => run('rebase_orders', mine, base, current)
