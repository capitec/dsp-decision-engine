// Differential fuzz, wasm side: replays fuzz.py's cases through the wasm core, compares with the Python reference's
// results after a JSON.parse/stringify round trip (JavaScript's own number and key-order view of them).
// node fuzz.mjs <cases.jsonl> <merge-core dir> (the wasm built into <merge-core dir>/pkg)
import { readFileSync } from 'node:fs'
const [file, dir] = process.argv.slice(2)
const core = await import(dir + '/js/merge-core.mjs')
await core.ready(readFileSync(dir + '/pkg/merge_core_bg.wasm'))
const { diff, apply, staleOps, conflicts, rebaseOrders, touched } = core
const S = JSON.stringify
const sortKeys = (v) => Array.isArray(v) ? v.map(sortKeys) : v && typeof v === 'object'
  ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortKeys(v[k])])) : v
function run(base, first, second, rops) {
  const out = {}
  const rec = (name, f) => {
    try { out[name] = S(f()) } catch (e) { out[name] = 'raises ' + String(e.message).split(':')[0] }
  }
  const d1 = diff(base, first), d2 = diff(base, second)
  out.diff1 = S(d1); out.diff2 = S(d2)
  rec('diff12', () => diff(first, second))
  const ap = (doc, ops) => { const [r, dr] = apply(doc, ops); return [r, dr.map((d) => ops.indexOf(d))] }
  rec('apply1', () => ap(base, d1))
  rec('apply_rand', () => ap(base, rops))
  rec('apply_rand_first', () => ap(first, rops))
  const stale = (mine, theirs) => staleOps(mine, theirs).map((x) => mine.indexOf(x))
  rec('stale', () => stale(d2, d1))
  rec('stale_rand', () => stale(rops, d1))
  rec('stale_rand_rev', () => stale(d1, rops))
  rec('online', () => {
    const doc = apply(base, d1)[0]
    const st = staleOps(d2, d1)
    return apply(doc, d2.filter((o) => !st.includes(o)))[0]
  })
  rec('conflicts', () => conflicts(d2, d1, base))
  rec('conflicts_rand', () => conflicts(rops, d1, base))
  rec('conflicts_rand_rev', () => conflicts(d1, rops, base))
  rec('rebase', () => rebaseOrders(d2, base, first))
  rec('rebase_rand', () => rebaseOrders(rops, base, first))
  rec('offline_merge', () => apply(first, rebaseOrders(d2, base, first))[0])
  rec('touched_rand', () => touched(rops))
  return out
}
// The deliberate change: conflicts reads only ASCII digits as an index (see fuzz.py).
function asciiOnly(want, got) {
  const a = JSON.parse(want), b = JSON.parse(got)
  const extra = a.filter((p) => !b.includes(p))
  return S(a.filter((p) => b.includes(p))) === S(b) &&
    extra.every((p) => p.split('/').some((x) => /[^\x00-\x7f]/.test(x) && /^[\p{Nd}\p{No}]+$/u.test(x)))
}
const counts = {}, mism = []
let n = 0
for (const line of readFileSync(file, 'utf8').split('\n')) {
  if (!line) continue
  const c = JSON.parse(line)
  const got = run(c.base, c.first, c.second, c.rops)
  n++
  for (const [k, refText] of Object.entries(c.ref)) {
    counts[k] = (counts[k] || 0) + 1
    const want = refText.startsWith('raises ') ? refText : S(JSON.parse(refText))
    if (want === got[k]) continue
    const keyOrderOnly = !want.startsWith('raises ') && !got[k].startsWith('raises ') && S(sortKeys(JSON.parse(want))) === S(sortKeys(JSON.parse(got[k])))
    const ra = want.startsWith('raises '), rb = got[k].startsWith('raises ')
    const tag = (ra && rb ? 'ERRCLASS ' : rb ? 'RUST-RAISES ' : !ra && k.startsWith('conflicts') && asciiOnly(want, got[k])
      ? 'ASCII-INDEX ' : keyOrderOnly ? 'KEYORDER ' : 'MISMATCH ') + k
    if (tag.startsWith('ASCII-INDEX')) { counts[tag] = (counts[tag] || 0) + 1; continue }
    if (rb) { counts[tag] = (counts[tag] || 0) + 1; continue }
    counts[tag] = (counts[tag] || 0) + 1
    if (mism.length < 50 && !keyOrderOnly) mism.push({ n, tree: c.tree, k, want: want.slice(0, 300), got: got[k].slice(0, 300) })
  }
}
console.log('cases', n)
for (const k of Object.keys(counts).sort()) console.log(k, counts[k])
for (const m of mism.slice(0, 15)) console.log(S(m))
