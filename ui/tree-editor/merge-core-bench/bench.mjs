// Time a JS adapter against the reference one, interleaved in one process, on out/bench_inputs.json.
//
//   node bench.mjs adapters/<approach>.mjs [--ref adapters/pure_ts.mjs] [-n 30] [--json out.json]
//
// The same tasks as bench.py; prints and writes median and p95 in ms per adapter.
import { readFileSync, writeFileSync } from 'node:fs'
import { basename, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const args = process.argv.slice(2)
const option = (flag, fallback) => (args.includes(flag) ? args[args.indexOf(flag) + 1] : fallback)
const n = Number(option('-n', 30))
const jsonOut = option('--json', null)
const names = [...new Set([resolve(args[0]), resolve(option('--ref', new URL('adapters/pure_ts.mjs', import.meta.url).pathname))])]
const adapters = {}
for (const name of names) {
  adapters[name] = await import(pathToFileURL(name).href)
  await adapters[name].ready
}
const inputs = JSON.parse(readFileSync(new URL('out/bench_inputs.json', import.meta.url), 'utf8'))

function tasks(m, d) {
  const state = m.load(d.base)
  return {
    load: () => m.load(d.base),
    export: () => m.export(state),
    roundtrip_one_edit: () => m.roundtrip(d.base, d.one_edit),
    roundtrip_50_edits: () => m.roundtrip(d.base, d.batch50),
    offline_50_vs_50: () => m.offline(d.base, d.mine50, d.theirs50),
    online_1_vs_1: () => m.online(d.base, d.first, d.second),
  }
}

function summary(times) {
  const sorted = [...times].sort((a, b) => a - b)
  const mid = sorted.length / 2
  const median = sorted.length % 2 ? sorted[Math.floor(mid)] : (sorted[mid - 1] + sorted[mid]) / 2
  const round = (ms) => Math.round(ms * 1e3) / 1e3
  return { median_ms: round(median), p95_ms: round(sorted[Math.ceil(0.95 * sorted.length) - 1]), n: sorted.length }
}

const results = {}
for (const [docId, d] of Object.entries(inputs)) {
  const runs = Object.fromEntries(names.map((name) => [name, tasks(adapters[name], d)]))
  const times = Object.fromEntries(names.map((name) => [name, Object.fromEntries(Object.keys(runs[name]).map((t) => [t, []]))]))
  for (let rep = 0; rep < n + 2; rep++) {
    // The first two are warm-up; alternate which adapter goes first, so neither always runs on a warmer cache.
    for (const task of Object.keys(runs[names[0]])) {
      for (const name of rep % 2 ? names : [...names].reverse()) {
        if (times[name][task] === null) continue
        const started = performance.now()
        try {
          runs[name][task]()
        } catch (e) {
          if (!String(e?.message).startsWith('not implemented')) throw e
          times[name][task] = null
          continue
        }
        if (rep >= 2) times[name][task].push(performance.now() - started)
      }
    }
  }
  for (const name of names) {
    const stem = basename(name).replace(/\.m?js$/, '')
    ;(results[stem] ??= {})[docId] = Object.fromEntries(
      Object.entries(times[name]).map(([t, v]) => [t, v ? summary(v) : 'not implemented']),
    )
  }
}
for (const [adapter, docs] of Object.entries(results)) {
  for (const [docId, byTask] of Object.entries(docs)) {
    for (const [task, s] of Object.entries(byTask)) {
      const line = typeof s === 'string' ? s : `median ${s.median_ms.toFixed(3).padStart(9)} ms  p95 ${s.p95_ms.toFixed(3).padStart(9)} ms`
      console.log(`${adapter.padEnd(14)} ${docId.padEnd(28)} ${task.padEnd(20)} ${line}`)
    }
  }
}
if (jsonOut) writeFileSync(jsonOut, JSON.stringify(results, null, 1))
