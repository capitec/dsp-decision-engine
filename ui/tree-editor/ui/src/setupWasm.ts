// The wasm of merge-core is initialised before the tests run, as the app does before rendering.
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { initSync } from '../../merge-core/pkg/merge_core.js'

initSync({ module: readFileSync(resolve(import.meta.dirname, '../../merge-core/pkg/merge_core_bg.wasm')) })
