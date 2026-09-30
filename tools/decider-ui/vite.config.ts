import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The standalone render harness: bundles demo/main.tsx into a static page that can be opened
// from a file:// path or served, so layoutlens and the vision model can audit the UI.
export default defineConfig({
  plugins: [react()],
  root: "demo",
  base: "./",
  build: {
    outDir: "../demo-dist",
    emptyOutDir: true,
    sourcemap: true,
  },
});
