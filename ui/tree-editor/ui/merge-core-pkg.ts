import { existsSync } from 'node:fs'

// Without this, a checkout without the wasm fails deep in the import graph with "Could not resolve ...".
export function requireMergeCorePkg() {
  if (!existsSync(new URL('../merge-core/pkg/merge_core.js', import.meta.url))) {
    throw new Error(
      'merge-core/pkg is missing: run `pnpm build:wasm` first (needs Rust, the wasm32-unknown-unknown target and ' +
        'wasm-pack; see ui/tree-editor/README.md, merge-core)',
    )
  }
}
