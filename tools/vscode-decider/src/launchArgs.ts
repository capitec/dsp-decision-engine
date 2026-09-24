import type { DebugProtocol } from "@vscode/debugprotocol";
import type { Controls } from "@decider/ui";

export interface AdapterOptions {
  python: string[];
  debugpyLibs?: string;
}

export interface LaunchArgs extends DebugProtocol.LaunchRequestArguments {
  program: string;
  pipeline?: string;
  data?: unknown;
  params?: unknown;
  /** Show this record's values instead of batch previews. */
  record?: number;
  stopOnEntry?: boolean;
  cwd?: string;
  /** Branch and loop forces and value breakpoints, as the flow panel set them. */
  controls?: Controls;
}

/** The Python to run the bridge with, for the standalone adapter: `DECIDER_PYTHON`, e.g. "uv run python". */
export function optionsFromEnv(): AdapterOptions {
  return { python: (process.env.DECIDER_PYTHON ?? "python3").split(" "), debugpyLibs: process.env.DECIDER_DEBUGPY_LIBS };
}
