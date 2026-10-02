import { builtinModules } from "node:module";
import { defineConfig } from "vitest/config";

// The extension host bundle: CommonJS, node target, `vscode` left external.
export default defineConfig({
  build: {
    lib: {
      entry: {
        extension: "src/extension.ts",
        adapterMain: "src/adapterMain.ts",
        runVsCodeTests: "test/vscode/runTests.ts",
        "vscode/index": "test/vscode/index.ts",
      },
      formats: ["cjs"],
      fileName: (_fmt, name) => `${name}.js`,
    },
    rollupOptions: {
      // vega-embed (reached via @decider/ui's App, for the experiment report) is browser-only: the extension
      // host never renders it, and vega-canvas's top-level `await import("canvas")` can't bundle for cjs anyway.
      external: ["vscode", "mocha", "vega", "vega-lite", "vega-embed", ...builtinModules, ...builtinModules.map((m) => `node:${m}`)],
    },
    outDir: "dist",
    emptyOutDir: false,
    sourcemap: true,
    target: "node18",
    minify: false,
    // A node build: without this Vite honours packages' `browser` field and
    // swaps @vscode/debugadapter's stdio runner for an empty stub.
    ssr: true,
  },
  ssr: { noExternal: true, target: "node" },
  test: {
    include: process.env.E2E ? [`test/e2e/${process.env.E2E === "1" ? "flows" : process.env.E2E}.e2e.ts`] : ["test/*.test.ts"],
    fileParallelism: false,
    testTimeout: 30000,
  },
});
