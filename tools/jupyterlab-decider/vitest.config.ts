import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/e2e/*.e2e.ts"],
    fileParallelism: false,
    testTimeout: 120_000,
    hookTimeout: 120_000,
  },
});
