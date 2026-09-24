import { Bridge } from "./bridge";
import { compareTraces, type Comparison, type TraceResult } from "./compare";
import type { Force } from "./protocol";

/** One side of a comparison: a pipeline file (maybe from another revision) and what to run it with. */
export interface Side {
  label: string;
  file: string;
  pipeline?: string;
  params?: unknown;
  overrides?: Record<string, unknown>;
  row?: number | null;
  data?: unknown;
  forces?: Force[];
}

/**
 * Run both sides to the end and line them up. `b` runs first so `a` can reuse
 * its input rows; both Python processes start at once to overlap the import.
 */
export async function runComparison(a: Side, b: Side, python: string[], cwd: string): Promise<Comparison> {
  const [ba, bb] = [new Bridge({ python, cwd }), new Bridge({ python, cwd })];
  try {
    const warm = ba.request("describe", { file: a.file, pipeline: a.pipeline }).catch(() => undefined);
    const tb = await bb.request<TraceResult>("trace", args(b, b.data));
    await warm;
    const ta = await ba.request<TraceResult>("trace", args(a, a.data ?? (a.overrides ? undefined : tb.data)));
    return compareTraces(ta, tb, a.label, b.label);
  } finally {
    await Promise.all([ba.dispose(), bb.dispose()]);
  }
}

function args(s: Side, data: unknown) {
  return { file: s.file, pipeline: s.pipeline, data: data ?? null, params: s.params ?? null, overrides: s.overrides ?? null, row: s.row ?? null, forces: s.forces ?? [] };
}
