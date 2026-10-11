// Cold load of a JS adapter in node: a fresh process per run, timed from the start of the import until the
// adapter is ready (its `ready` promise, when it exports one, resolved). Prints the median of N (20) runs in ms.
//
//   node coldload.mjs adapters/<approach>.mjs [-n 20]
import { execFileSync } from 'node:child_process'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const args = process.argv.slice(2)
const n = args.includes('-n') ? Number(args[args.indexOf('-n') + 1]) : 20
const url = pathToFileURL(resolve(args[0])).href
const code = `const t = performance.now(); const m = await import(${JSON.stringify(url)}); await m.ready;
console.log(performance.now() - t)`
const times = []
for (let i = 0; i < n; i++) {
  times.push(Number(execFileSync(process.execPath, ['--input-type=module', '-e', code], { encoding: 'utf8' })))
}
times.sort((a, b) => a - b)
const median = n % 2 ? times[(n - 1) / 2] : (times[n / 2 - 1] + times[n / 2]) / 2
console.log(JSON.stringify({ adapter: args[0], n, median_ms: Math.round(median * 1e3) / 1e3,
  min_ms: Math.round(times[0] * 1e3) / 1e3, max_ms: Math.round(times[n - 1] * 1e3) / 1e3 }))
