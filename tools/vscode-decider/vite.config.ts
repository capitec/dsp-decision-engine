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
      external: ["vscode", "mocha", ...builtinModules, ...builtinModules.map((m) => `node:${m}`)],
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
    include: process.env.E2E ? ["test/e2e/*.e2e.ts"] : ["test/*.test.ts"],
    fileParallelism: false,
    testTimeout: 30000,
  },
});
