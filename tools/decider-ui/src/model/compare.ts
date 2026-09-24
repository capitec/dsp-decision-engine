import { callNodes, formatValue, type CallNodeJson, type DescribeResult, type RecordKey } from "./protocol";

/** What every call of a run wrote, and its output if it finished. */
export interface RunTrace {
  steps: Record<string, Record<string, unknown[]>>;
  output: Record<string, unknown[]> | null;
  error: string | null;
}

/** One whole run, as the bridge's `trace` returns it. */
export interface TraceResult extends DescribeResult, RunTrace {
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
  /** "not taken": it ran in the first run only; the second sent every record another way (a branch arm, a loop that stopped). */
  status: "same" | "changed" | "added" | "removed" | "not run" | "not taken";
  /** What changed in the step itself: "code", "params", "reads", "writes". */
  structural: string[];
  /** Each changed param, as "cap: 48 → 42". */
  paramChanges: string[];
  outputs: ValueDiff[];
  /** Where the step is defined, as "policy.py:193". */
  where?: string;
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
  /** The params documents each run used, when they differ (a what-if or a scenario). */
  paramsDocs?: { a: unknown; b: unknown };
  /** Shared param -> the steps that read it. */
  sharedUsers?: Record<string, string[]>;
  /** The runs differ by a forced branch arm or loop count, not by an edit. */
  forced?: boolean;
  /** The records the force named; the others ran as they are. */
  forcedRows?: number[];
  /** Said under the title, e.g. how the two runs were made. */
  note?: string;
  /** The file the flow's PARAMS come from, e.g. "params.json". */
  valuesFile?: string;
  /** Each run's PARAMS document (tables' rows and tuned values), for saying what a param was. */
  values?: { a: unknown; b: unknown };
}

/** `values` (a PARAMS document) over the default of every param in `schema`, nested by path. */
export function docWithDefaults(schema: DescribeResult["params"], values: unknown): Record<string, unknown> {
  const doc: Record<string, unknown> = {};
  for (const [path, ps] of Object.entries(schema)) {
    let at = doc;
    for (const part of path.split("/")) at = (at[part] ??= {}) as Record<string, unknown>;
    for (const [k, info] of Object.entries(ps)) if (info.default !== undefined) at[k] = info.default;
  }
  const merge = (into: Record<string, unknown>, from: unknown) => {
    for (const [k, v] of Object.entries((from ?? {}) as Record<string, unknown>)) {
      if (v && typeof v === "object" && !Array.isArray(v) && into[k] && typeof into[k] === "object") merge(into[k] as Record<string, unknown>, v);
      else into[k] = v;
    }
  };
  merge(doc, values);
  return doc;
}

/** The parts of `b` that differ from `a`, nested the same way. */
export function diffDoc(a: unknown, b: unknown): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const isDoc = (x: unknown): x is Record<string, unknown> => !!x && typeof x === "object" && !Array.isArray(x);
  for (const [k, v] of Object.entries(isDoc(b) ? b : {})) {
    const was = isDoc(a) ? a[k] : undefined;
    if (isDoc(v) && isDoc(was)) {
      const inner = diffDoc(was, v);
      if (Object.keys(inner).length) out[k] = inner;
    } else if (!same(was, v)) out[k] = v;
  }
  return out;
}

/** "repo_rate: 0.0775 → 0.075", or for a table "pl_base_rates row 3: pl_base_rate 0.255 → 0.245". */
export function paramChangeLines(doc: unknown, before: unknown): string[] {
  const lines: string[] = [];
  const at = (d: unknown, k: string) => (d && typeof d === "object" ? (d as Record<string, unknown>)[k] : undefined);
  const visit = (d: unknown, was: unknown) => {
    for (const [k, v] of Object.entries((d ?? {}) as Record<string, unknown>)) {
      const old = at(was, k);
      if (Array.isArray(v)) {
        const rows = (Array.isArray(old) ? old : []) as Record<string, unknown>[];
        v.forEach((row, i) => {
          const r = row as Record<string, unknown>;
          if (!rows[i]) return lines.push(`${k} row ${i + 1} added`);
          const cells = Object.keys(r).filter((c) => !same(rows[i][c], r[c])).map((c) => `${c} ${formatValue(rows[i][c], c)} → ${formatValue(r[c], c)}`);
          const key = Object.keys(r).filter((c) => same(rows[i][c], r[c])).slice(0, 2).map((c) => `${c} ${formatValue(r[c], c)}`);
          if (cells.length) lines.push(`${k} row ${i + 1}${key.length ? ` (${key.join(", ")})` : ""}: ${cells.join(", ")}`);
        });
        if (rows.length > v.length) lines.push(`${k}: ${rows.length - v.length} row${rows.length - v.length === 1 ? "" : "s"} removed`);
      } else if (v && typeof v === "object") visit(v, old);
      else lines.push(`${k}: ${old === undefined ? "default" : formatValue(old, k)} → ${formatValue(v, k)}`);
    }
  };
  visit(doc, before);
  return lines;
}

/** Each param a params document sets, with the steps that read it. */
export function paramReaders(doc: unknown, sharedUsers: Record<string, string[]> = {}): { param: string; steps: string[] }[] {
  const out: { param: string; steps: string[] }[] = [];
  const visit = (d: unknown, path: string[]) => {
    for (const [k, v] of Object.entries((d ?? {}) as Record<string, unknown>)) {
      if (v && typeof v === "object" && !Array.isArray(v)) visit(v, [...path, k]);
      else out.push(path.length === 1 && path[0] === "shared" ? { param: k, steps: sharedUsers[k] ?? [] } : { param: k, steps: [path.join("/")] });
    }
  };
  visit(doc, []);
  return out;
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
  const where = (path: string) => {
    const n = nodesB.get(path) ?? nodesA.get(path);
    return n?.file ? `${n.file.slice(n.file.lastIndexOf("/") + 1)}${n.line ? `:${n.line}` : ""}` : undefined;
  };
  const steps: StepDiff[] = merged([...nodesA.values()], [...nodesB.values()]).map((path): StepDiff => ({ ...stepDiff(path), where: where(path) }));
  function stepDiff(path: string): StepDiff {
    const na = nodesA.get(path);
    const nb = nodesB.get(path);
    if (!na) return { path, status: "added", structural: [], paramChanges: [], outputs: [] };
    if (!nb) return { path, status: "removed", structural: [], paramChanges: [], outputs: [] };
    const sa = a.steps[path];
    const sb = b.steps[path];
    const changes = structural(na, nb);
    const params = paramChanges(na, nb);
    if (!sa && !sb) return { path, status: changes.length ? "changed" : "not run", structural: changes, paramChanges: params, outputs: [] };
    // A skipped step is gone from the IR (above); in both but run in one only, the other took another arm.
    if (sa && !sb) return { path, status: "not taken", structural: changes, paramChanges: params, outputs: [] };
    const outputs = diffColumns(sa ?? {}, sb ?? {});
    return { path, status: outputs.length || changes.length ? "changed" : "same", structural: changes, paramChanges: params, outputs };
  }
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
    values: { a: docWithDefaults(a.params ?? {}, a.values), b: docWithDefaults(b.params ?? {}, b.values) },
    valuesFile: b.valuesFile ?? undefined,
    // Two revisions each run with their own PARAMS; what differs between them is a param change.
    paramsDocs: Object.keys(diffDoc(a.values, b.values)).length ? { a: a.values, b: diffDoc(a.values, b.values) } : undefined,
    sharedUsers: Object.fromEntries(Object.entries(b.params?.shared ?? {}).map(([k, info]) => [k, (info.used_by as string[] | undefined) ?? []])),
  };
}
