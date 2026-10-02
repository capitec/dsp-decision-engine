import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The graph webview: one JS and one CSS file with fixed names, loaded by graphPanel.ts.
export default defineConfig({
  plugins: [react()],
  root: "webview",
  base: "./",
  build: {
    outDir: "../dist/webview",
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      input: "webview/main.tsx",
      output: {
        entryFileNames: "index.js",
        assetFileNames: "index.[ext]",
      },
    },
  },
});
