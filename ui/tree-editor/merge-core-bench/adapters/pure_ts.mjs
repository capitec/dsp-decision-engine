// Reference adapter: the pure TypeScript live ops as they were before any library replaced them (`reference/`),
// run with node's type stripping.
import { apply, conflicts, diff, isRelated, reconcile, staleOps, touched } from '../reference/live-ops.ts'

export { apply, conflicts, diff, isRelated, reconcile, staleOps, touched }

// The TypeScript copy has no rebaseOrders: only the server rebases.
export function rebaseOrders() {
  throw new Error('not implemented: the TypeScript copy has no rebaseOrders')
}

// Cases where the pure TypeScript copy differs from the pure Python one, which made the expected results.
const NO_REBASE = 'the TypeScript copy has no rebaseOrders, so its offline merge undoes their reorder'
export const XFAIL = {
  'offline: their reorder against my insert merges (only they changed the common keys\' order)': NO_REBASE,
  'offline: my insert onto their reorder keeps their order': NO_REBASE,
  'offline: fraud-overlay-base (rule ids stamped) pair 1': NO_REBASE,
  'integers past 2**53 that differ by one differ': 'a JavaScript number holds integers exactly only up to 2**53',
  'an integer past 2**53 and the float it rounds to differ': 'a JavaScript number holds integers exactly only up to 2**53',
  'an integer past 2**53 and the float it rounds to on the two sides conflict':
    'a JavaScript number holds integers exactly only up to 2**53',
  'an added integer-like key goes last': 'JavaScript objects list integer-like keys first, whatever the insertion order',
  'diff orders keys by code point':
    'Array.prototype.sort orders by UTF-16 code unit, so an astral character sorts before U+E000..U+FFFF',
  'touched of an op without a path raises': 'touched does not check ops; it returns undefined for the path',
  'touched of an op that is not an object raises': 'touched does not check ops; it returns undefined for the path',
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
  const found = conflicts(diff(base, mine), diff(base, theirs), base)
  if (found.length) return { conflicts: found }
  return { merged: apply(theirs, diff(base, mine))[0] }
}

export const load = (doc) => doc
const exportDoc = (state) => state
export { exportDoc as export }
