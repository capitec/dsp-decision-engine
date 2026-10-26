import { builtinModules } from "node:module";
import { defineConfig } from "vitest/config";

// The extension host bundle: CommonJS, node target, `vscode` left external.
export default defineConfig({
  build: {
    lib: {
      entry: { extension: "src/extension.ts" },
      formats: ["cjs"],
      fileName: (_fmt, name) => `${name}.js`,
    },
    rollupOptions: {
      external: ["vscode", "vega", "vega-lite", "vega-embed", ...builtinModules, ...builtinModules.map((m) => `node:${m}`)],
    },
    outDir: "dist",
    emptyOutDir: false,
    sourcemap: true,
    target: "node18",
    minify: false,
    ssr: true,
  },
  ssr: { noExternal: true, target: "node" },
  test: { include: ["test/*.test.ts"] },
});
