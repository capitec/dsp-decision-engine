import { callNodes, type CallNodeJson, type DescribeResult, type RecordKey } from "./protocol";

/** One whole run, as the bridge's `trace` returns it. */
export interface TraceResult extends DescribeResult {
  steps: Record<string, Record<string, unknown[]>>;
  output: Record<string, unknown[]> | null;
  error: string | null;
  data: Record<string, unknown>[];
  key?: RecordKey;
}

export interface ValueDiff {
  name: string;
  changedRows: number[];
  samples: { row: number; a: unknown; b: unknown }[];
}

export interface StepDiff {
  path: string;
  status: "same" | "changed" | "added" | "removed" | "not run";
  /** What changed in the step itself: "code", "params", "reads", "writes". */
  structural: string[];
  /** Each changed param, as "cap: 48 → 42". */
  paramChanges: string[];
  outputs: ValueDiff[];
}

export interface Comparison {
  a: string;
  b: string;
  steps: StepDiff[];
  output: ValueDiff[];
  /** The first step, in execution order, whose outputs differ. */
  firstDivergence: string | null;
  errors: { a: string | null; b: string | null };
  rows: number;
  key: RecordKey;
  /** Every output column, so a view can say which stayed the same. */
  outputColumns: string[];
  /** Columns that were pipeline inputs rather than results. */
  inputColumns: string[];
  /** Both runs' results (not inputs), for showing values that did not change too. */
  results: { a: Record<string, unknown[]>; b: Record<string, unknown[]> };
  /** Inputs the second run changed on purpose: a what-if's or a scenario's own values. */
  changedInputs: { name: string; after: unknown; scope: string }[];
  /** The two pipeline files, when they differ (a revision against the working tree). */
  files?: { a: string; b: string };
}

const SAMPLES = 3;

export function same(a: unknown, b: unknown): boolean {
  if (typeof a === "number" && typeof b === "number") {
    if (Number.isNaN(a) && Number.isNaN(b)) return true;
    return Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));
  }
  return JSON.stringify(a) === JSON.stringify(b);
}

function diffColumns(a: Record<string, unknown[]>, b: Record<string, unknown[]>): ValueDiff[] {
  const out: ValueDiff[] = [];
  for (const name of [...new Set([...Object.keys(b), ...Object.keys(a)])]) {
    const av = a[name] ?? [];
    const bv = b[name] ?? [];
    const changedRows: number[] = [];
    for (let r = 0; r < Math.max(av.length, bv.length); r++) if (!same(av[r], bv[r])) changedRows.push(r);
    if (changedRows.length) out.push({ name, changedRows, samples: changedRows.slice(0, SAMPLES).map((row) => ({ row, a: av[row], b: bv[row] })) });
  }
  return out;
}

function paramChanges(a: CallNodeJson, b: CallNodeJson): string[] {
  return [...new Set([...Object.keys(a.params), ...Object.keys(b.params)])]
    .filter((k) => !same(a.params[k], b.params[k]))
    .map((k) => `${k}: ${fmt(a.params[k])} → ${fmt(b.params[k])}`);
}

const fmt = (v: unknown) => (v === undefined ? "none" : typeof v === "number" ? String(Number(v.toFixed(4))) : JSON.stringify(v));

function structural(a: CallNodeJson, b: CallNodeJson): string[] {
  const changes: string[] = [];
  if (a.code !== b.code) changes.push("code");
  if (JSON.stringify(a.params) !== JSON.stringify(b.params)) changes.push("params");
  if (JSON.stringify(a.inputs) !== JSON.stringify(b.inputs)) changes.push("reads");
  if (JSON.stringify(a.outputs) !== JSON.stringify(b.outputs)) changes.push("writes");
  return changes;
}

/** B's call nodes in execution order, with A-only nodes slotted in after their A predecessor. */
function merged(a: CallNodeJson[], b: CallNodeJson[]): string[] {
  const order = b.map((n) => n.path);
  const inB = new Set(order);
  let after: string | null = null;
  for (const n of a) {
    if (inB.has(n.path)) {
      after = n.path;
      continue;
    }
    const at: number = after === null ? 0 : order.indexOf(after) + 1;
    order.splice(at, 0, n.path);
    after = n.path;
  }
  return order;
}

/**
 * Line two runs up step by step: what changed in each step, and which of its
 * outputs differ on which rows. `a` is the baseline.
 *
 * Example::
 *
 *     compareTraces(atTag, workingTree, "v1.2", "working tree").firstDivergence
 */
export function compareTraces(a: TraceResult, b: TraceResult, labelA: string, labelB: string): Comparison {
  const nodesA = new Map(callNodes(a.ir).map((n) => [n.path, n]));
  const nodesB = new Map(callNodes(b.ir).map((n) => [n.path, n]));
  const steps: StepDiff[] = merged([...nodesA.values()], [...nodesB.values()]).map((path) => {
    const na = nodesA.get(path);
    const nb = nodesB.get(path);
    if (!na) return { path, status: "added", structural: [], paramChanges: [], outputs: [] };
    if (!nb) return { path, status: "removed", structural: [], paramChanges: [], outputs: [] };
    const sa = a.steps[path];
    const sb = b.steps[path];
    const changes = structural(na, nb);
    const params = paramChanges(na, nb);
    if (!sa && !sb) return { path, status: changes.length ? "changed" : "not run", structural: changes, paramChanges: params, outputs: [] };
    const outputs = diffColumns(sa ?? {}, sb ?? {});
    return { path, status: outputs.length || changes.length ? "changed" : "same", structural: changes, paramChanges: params, outputs };
  });
  const inputs = new Set(Object.keys(b.data[0] ?? a.data[0] ?? {}));
  const results = (o: Record<string, unknown[]> | null) => Object.fromEntries(Object.entries(o ?? {}).filter(([k]) => !inputs.has(k)));
  const changedInputs = Object.keys(b.data[0] ?? {})
    .map((name) => {
      const rows = b.data.map((_, r) => r).filter((r) => !same(a.data[r]?.[name], b.data[r]?.[name]));
      return rows.length ? { name, after: b.data[rows[0]][name], scope: rows.length === b.data.length ? "every record" : rows.map((r) => (b.key ? `${b.key.name} ${fmt(b.key.values[r])}` : `row ${r}`)).join(", ") } : null;
    })
    .filter((x): x is { name: string; after: unknown; scope: string } => x !== null);
  return {
    a: labelA,
    b: labelB,
    steps,
    inputColumns: [...inputs],
    results: { a: results(a.output), b: results(b.output) },
    changedInputs,
    output: a.output && b.output ? diffColumns(results(a.output), results(b.output)) : [],
    firstDivergence: steps.find((s) => s.outputs.length)?.path ?? null,
    errors: { a: a.error, b: b.error },
    rows: b.data.length,
    key: b.key ?? null,
    outputColumns: Object.keys(b.output ?? a.output ?? {}),
  };
}
