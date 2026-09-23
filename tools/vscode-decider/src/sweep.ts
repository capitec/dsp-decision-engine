import { compareTraces, same, type Comparison } from "./compare";
import type { DescribeResult } from "./protocol";

export interface RunTrace {
  steps: Record<string, Record<string, unknown[]>>;
  output: Record<string, unknown[]> | null;
  error: string | null;
}

/** What the bridge's `sweep` returns: the unchanged continuation and one run per scenario. */
export interface SweepResponse {
  baseline: RunTrace;
  results: (RunTrace & { label: string })[];
  describe: DescribeResult;
  data: Record<string, unknown>[];
  at: { path: string; when: string; n: number } | null;
}

/** A scenario as the bridge runs it. */
export interface Scenario {
  label: string;
  params?: Record<string, unknown>;
  overrides?: Record<string, unknown>;
  row?: number | null;
}

/** One knob of a sweep: a param (`path|name`) or a value by column name, and the values to try. */
export interface Knob {
  kind: "param" | "value";
  key: string;
  values: unknown[];
}

export interface Sweep {
  /** Where the scenarios fork: "before term/cap_by_income", or null for the start. */
  at: string | null;
  labels: string[];
  base: Record<string, unknown[]> | null;
  outputs: (Record<string, unknown[]> | null)[];
  errors: (string | null)[];
  /** Each scenario against the unchanged run, step by step. */
  comparisons: Comparison[];
  /** Output columns that differ in at least one scenario, in output order. */
  changedColumns: string[];
  rows: number;
}

/** Every combination of the knobs' values, as scenarios. */
export function scenarios(knobs: Knob[], row: number | null): Scenario[] {
  let combos: [Knob, unknown][][] = [[]];
  for (const k of knobs) combos = combos.flatMap((c) => k.values.map((v) => [...c, [k, v] as [Knob, unknown]]));
  return combos
    .filter((c) => c.length)
    .map((c) => {
      const sc: Scenario = { label: c.map(([k, v]) => `${k.key.replace("|", ".")}=${JSON.stringify(v)}`).join(", "), row };
      for (const [k, v] of c) {
        if (k.kind === "value") (sc.overrides ??= {})[k.key] = v;
        else {
          const [path, name] = k.key.split("|");
          let target = (sc.params ??= {}) as Record<string, unknown>;
          for (const part of path.split("/")) target = (target[part] ??= {}) as Record<string, unknown>;
          target[name] = v;
        }
      }
      return sc;
    });
}

export function summariseSweep(r: SweepResponse): Sweep {
  const asTrace = (t: RunTrace) => ({ ...r.describe, ...t, data: r.data });
  const base = r.baseline.output;
  const changed = new Set<string>();
  for (const res of r.results)
    for (const [name, values] of Object.entries(res.output ?? {}))
      if (!base?.[name] || values.some((v, i) => !same(v, base[name][i]))) changed.add(name);
  return {
    at: r.at ? `${r.at.when} ${r.at.path}${r.at.n > 1 ? ` (time ${r.at.n})` : ""}` : null,
    labels: r.results.map((x) => x.label),
    base,
    outputs: r.results.map((x) => x.output),
    errors: r.results.map((x) => x.error),
    comparisons: r.results.map((x) => compareTraces(asTrace(r.baseline), asTrace(x), "original", x.label)),
    changedColumns: Object.keys(base ?? r.results[0]?.output ?? {}).filter((c) => changed.has(c)),
    rows: r.data.length,
  };
}
