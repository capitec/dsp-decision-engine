import { compareTraces, same, type Comparison } from "./compare";
import type { DescribeResult, RecordKey } from "./protocol";

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
  key: RecordKey;
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
  key: RecordKey;
  /** Per scenario, the knob values it used, for one column per knob. */
  knobs: { name: string; values: unknown[] }[];
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
  /** Each knob's value in the original run, per record (a param's default is the same on every row). */
  knobBase: Record<string, unknown[]>;
  rows: number;
}

/** Every combination of the knobs' values, as scenarios. */
export function scenarios(knobs: Knob[], row: number | null): Scenario[] {
  let combos: [Knob, unknown][][] = [[]];
  for (const k of knobs) combos = combos.flatMap((c) => k.values.map((v) => [...c, [k, v] as [Knob, unknown]]));
  return combos
    .filter((c) => c.length)
    .map((c) => {
      const sc: Scenario = { label: c.map(([k, v]) => `${k.key.replace("|", " · ")} = ${JSON.stringify(v)}`).join(", "), row };
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

/** `{"term/cap_by_income · cap": 24, "requested_amount": 50000}` for one scenario. */
export function knobValues(s: Scenario): Record<string, unknown> {
  const out: Record<string, unknown> = { ...(s.overrides ?? {}) };
  const walk = (doc: Record<string, unknown>, prefix: string) => {
    for (const [k, v] of Object.entries(doc)) {
      if (v && typeof v === "object" && !Array.isArray(v)) walk(v as Record<string, unknown>, prefix ? `${prefix}/${k}` : k);
      else out[`${prefix} · ${k}`] = v;
    }
  };
  walk(s.params ?? {}, "");
  return out;
}

export function summariseSweep(r: SweepResponse, scenarioList: Scenario[] = []): Sweep {
  const asTrace = (t: RunTrace) => ({ ...r.describe, ...t, data: r.data, key: r.key });
  const base = r.baseline.output;
  const changed = new Set<string>();
  for (const res of r.results)
    for (const [name, values] of Object.entries(res.output ?? {}))
      if (!base?.[name] || values.some((v, i) => !same(v, base[name][i]))) changed.add(name);
  // In the order the user listed the knobs, which is the order of the scenario labels.
  const firstLabel = scenarioList[0]?.label ?? "";
  const knobNames = [...new Set(scenarioList.flatMap((s) => Object.keys(knobValues(s))))].sort(
    (a, b) => firstLabel.indexOf(a) - firstLabel.indexOf(b),
  );
  const inputs = new Set(Object.keys(r.data[0] ?? {}));
  const knobBase = Object.fromEntries(
    knobNames.map((name) => {
      if (inputs.has(name)) return [name, r.data.map((row) => row[name])];
      const [path, param] = name.split(" · ");
      return [name, r.data.map(() => r.describe.params[path]?.[param]?.default)];
    }),
  );
  // A fork sets inputs mid-run, so its output still shows the input as it arrived; name the change instead.
  const withInputs = (c: Comparison, s: Scenario | undefined): Comparison => ({
    ...c,
    paramsDocs: { a: {}, b: s?.params ?? {} },
    changedInputs: [
      ...Object.entries(knobValues({ label: "", params: s?.params }))
        .map(([name, after]) => ({ name, after, scope: "steps after the pause" })),
      ...Object.entries(s?.overrides ?? {}).map(([name, after]) => ({
        name,
        after,
        scope: s?.row == null ? "every record" : r.key ? `${r.key.name} ${String(r.key.values[s.row])}` : `row ${s.row}`,
      })),
    ],
  });
  return {
    key: r.key,
    knobs: knobNames.map((name) => ({ name, values: scenarioList.map((s) => knobValues(s)[name]) })),
    at: r.at ? `${r.at.when} ${r.at.path}${r.at.n > 1 ? ` (time ${r.at.n})` : ""}` : null,
    labels: r.results.map((x) => x.label),
    base,
    outputs: r.results.map((x) => x.output),
    errors: r.results.map((x) => x.error),
    comparisons: r.results.map((x, i) => withInputs(compareTraces(asTrace(r.baseline), asTrace(x), "original", x.label), scenarioList[i])),
    knobBase,
    changedColumns: Object.keys(base ?? r.results[0]?.output ?? {}).filter((c) => changed.has(c) && !inputs.has(c)),
    rows: r.data.length,
  };
}
