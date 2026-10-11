// Cost of crossing into the wasm core as JSON text against serde-wasm-bindgen (needs ../pkg-direct).
//   node boundary.mjs ../../merge-core-bench/out/bench_inputs.json [n]
import { readFileSync } from 'node:fs'
import { performance } from 'node:perf_hooks'
import { callDirect, call, initSync } from '../pkg-direct/merge_core.js'

initSync({ module: readFileSync(new URL('../pkg-direct/merge_core_bg.wasm', import.meta.url)) })
const inputs = JSON.parse(readFileSync(process.argv[2], 'utf8'))
const n = Number(process.argv[3] ?? 30)

function med(f) {
  const times = []
  for (let i = 0; i < n; i++) {
    const t = performance.now()
    f()
    times.push(performance.now() - t)
  }
  return Math.round(times.sort((a, b) => a - b)[n >> 1] * 1000) / 1000
}

for (const [name, d] of Object.entries(inputs)) {
  const ops = JSON.parse(call('diff', JSON.stringify([d.base, d.batch50])))
  const args = [d.base, ops]
  const text = JSON.stringify(args)
  const empty = JSON.stringify([d.base, []])
  const out = call('apply', text)
  console.log(name, {
    'wasm parse + clone + serialize only (no ops)': med(() => call('apply', empty)),
    'JSON.stringify of args': med(() => JSON.stringify(args)),
    'wasm apply on text (parse + apply + serialize)': med(() => call('apply', text)),
    'JSON.parse of result': med(() => JSON.parse(out)),
    'text route total': med(() => JSON.parse(call('apply', JSON.stringify(args)))),
    'serde-wasm-bindgen route total': med(() => callDirect('apply', args)),
  })
}

// A __proto__ key must stay data.
const proto = JSON.parse('{"__proto__": {"polluted": true}, "a": 1}')
for (const [label, f] of [['text', () => JSON.parse(call('diff', JSON.stringify([{}, proto])))], ['direct', () => callDirect('diff', [{}, proto])]]) {
  try {
    const ops = f()
    console.log('proto', label, JSON.stringify(ops), 'Object.prototype.polluted =', ({}).polluted)
  } catch (e) {
    console.log('proto', label, 'threw', e.message)
  }
}
