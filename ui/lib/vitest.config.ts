import { defineConfig } from "vitest/config";

// The tests start the Python bridge, which takes seconds.
export default defineConfig({ test: { testTimeout: 30000 } });
