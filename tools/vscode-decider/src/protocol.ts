import type { Comparison } from "./compare";
import type { Scenario, Sweep } from "./sweep";

// Shapes the Python bridge sends. Shared by the adapter, the tree and the webview.

export interface IRNodeBase {
  path: string;
  source: string;
  file: string | null;
  line: number | null;
}

export interface CallNodeJson extends IRNodeBase {
  kind: "call";
  callKind: "scalar" | "row" | "frame";
  /** `null`: a frame step of unknown lineage. */
  inputs: string[] | null;
  outputs: string[] | null;
  params: Record<string, unknown>;
  /** The Python that runs in interpreted mode: `fn`, or a row node's `reference`. */
  python: { file: string; line: number; bodyLine: number | null } | null;
  /** Changes when the code or config behind the node changes. */
  code: string;
  /** The first line of the step's docstring. */
  doc?: string;
  /** What a one-line step returns, e.g. `min(pl_raw_rate, repo_rate + cap_margin)`. */
  formula?: string | null;
  /** Set once the step's code was edited mid-run: the formula it had before. */
  formulaBefore?: string | null;
  /** A lookup table's match, e.g. `{type: "between", variable, lower_bound_column, upper_bound_column}`. */
  table?: Record<string, unknown> | null;
}

export interface GroupNodeJson extends IRNodeBase {
  kind: "sequence" | "branch" | "loop";
  modifies?: string[];
  carries?: string[];
  maxIterations?: number;
  children: IRNodeJson[];
  /** Set on a group drawn folded: the paths of the steps inside it. */
  folded?: string[];
}

export type IRNodeJson = CallNodeJson | GroupNodeJson;

export interface DescribeResult {
  pipelines: { name: string; line: number | null; kind: string }[];
  pipeline: string;
  ir: IRNodeJson;
  /** Node path (or "shared") -> param name -> its type, default and bounds. */
  params: Record<string, Record<string, ParamInfo>>;
  /** The module's `PARAMS` document: the values the flow runs with (tables' rows included). */
  values?: Record<string, unknown>;
}

export interface ParamInfo {
  type: string;
  default?: unknown;
  required?: boolean;
  ge?: number;
  le?: number;
  gt?: number;
  lt?: number;
  [k: string]: unknown;
}

export interface Checkpoint {
  path: string;
  when: "before" | "after";
}

export interface Summary {
  dtype: string;
  rows: number;
  nulls: number;
  preview: unknown[];
}

export interface SessionEvent {
  kind: string;
  origin?: { path: string; source: string; locator: string | null };
  [k: string]: unknown;
}

export interface Status {
  finished: boolean;
  current: Checkpoint | null;
  events: SessionEvent[];
  error?: string;
}

export interface ColumnSummary extends Summary {
  name: string;
  /** The focused record's value, when a record is focused. */
  value: unknown;
  producer: string;
  versions: number;
}

export interface Lineage {
  name: string;
  producer: string | null;
  value: unknown;
  via?: "merge" | "carry";
  inputs: Lineage[];
}

/** Tree positions a row node reached in its latest run: locator -> rows. */
export type Visits = Record<string, Record<string, number>>;

export interface RunStatus {
  current: Checkpoint | null;
  finished: boolean;
  finishedPaths: string[];
  visits: Visits;
  record: number | null;
  /** Steps edited in this run: path -> "delete" (skipped) or "replace" (swapped for edited code). */
  edits?: Record<string, "delete" | "replace">;
}

export interface ColumnHistory {
  name: string;
  /** `written` is false when a version never reached the focused record (a branch arm it did not take). */
  versions: { producer: string; values: unknown[]; written?: boolean }[];
}

/** The column that names a record, and its value on every row. */
export type RecordKey = { name: string; values: unknown[] } | null;

/** "client_id 2", or "row 1" when the data has no id column. */
export function recordLabel(row: number, key: RecordKey): string {
  return key ? `${key.name} ${String(key.values[row])}` : `row ${row}`;
}

/** Values for people: no float noise; amounts of 100 or more to two decimals (58113.07), smaller ones to four. */
/** Names whose values read as percentages: rates, loadings, discounts, margins. */
export const isRateName = (name?: string) => !!name && /(rate|loading|discount|margin)s?$/.test(name);
/** Names whose values are rand amounts. */
export const isMoneyName = (name?: string) => !!name && /(amount|cost|income|fee|instalment|expenses)s?$/.test(name);

/** A value as the UI shows it; with its column's `name`, a rate below 1 shows as a percentage ("25.2%"). */
export function formatValue(v: unknown, name?: string): string {
  if (v === undefined) return "—";
  if (v === null) return "empty";
  if (typeof v === "number" && isRateName(name) && Math.abs(v) < 1) return `${Number((v * 100).toFixed(3))}%`;
  if (typeof v === "number" && isMoneyName(name)) return `R\u00a0${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  if (typeof v === "number") return v.toLocaleString("en-US", { maximumFractionDigits: Math.abs(v) >= 100 ? 2 : 4 });
  return typeof v === "string" ? v : JSON.stringify(v);
}

export type Tab = "graph" | "state" | "params" | "scenarios" | "compare";

/** Messages between the extension and the graph webview. */
export type ToWebview =
  | { type: "describe"; describe: DescribeResult }
  | ({ type: "status" } & RunStatus)
  | { type: "state"; columns: ColumnSummary[] | null; rows: number; key: RecordKey }
  | { type: "lineage"; lineage: Lineage | null; history: ColumnHistory | null }
  | { type: "treePath"; path: string; row: number; visited: string[]; result?: unknown[] }
  | { type: "compare"; comparison: Comparison | null; busy?: string; error?: string }
  | { type: "tab"; tab: Tab }
  | { type: "select"; path: string }
  | { type: "edited"; path: string; diff: string[]; formula: string | null }
  | { type: "sweep"; sweep: Sweep | null; busy?: string; error?: string };

export type FromWebview =
  | { type: "ready" }
  | { type: "reveal"; path: string }
  | { type: "record"; row: number | null }
  | { type: "lineage"; name: string }
  | { type: "treePath"; path: string }
  | { type: "rewind"; path: string }
  | { type: "skip"; path: string }
  | { type: "reloadStep"; path: string }
  | { type: "compareEdits"; label: string; edits: Record<string, "delete" | "replace">; path?: string }
  | { type: "whatIf"; params: unknown; overrides: Record<string, unknown>; row: number | null; label: string }
  | { type: "restartWith"; params: unknown }
  | { type: "compareRevision" }
  | { type: "runTo"; path: string }
  | { type: "maximise" }
  | { type: "step" }
  | { type: "openDiff"; path: string }
  | { type: "sweep"; scenarios: Scenario[]; fromHere: boolean };

export function walk(node: IRNodeJson, fn: (n: IRNodeJson, parent: IRNodeJson | null) => void, parent: IRNodeJson | null = null) {
  fn(node, parent);
  if (node.kind !== "call") for (const c of node.children) walk(c, fn, node);
}

export function callNodes(node: IRNodeJson): CallNodeJson[] {
  const out: CallNodeJson[] = [];
  walk(node, (n) => {
    if (n.kind === "call") out.push(n);
  });
  return out;
}

export function lastSegment(path: string): string {
  return path === "" ? "<root>" : path.slice(path.lastIndexOf("/") + 1);
}

export function kindLabel(n: IRNodeJson): string {
  return n.kind === "call" ? n.callKind : n.kind;
}

export function previewOf(c: Summary, name?: string): string {
  // Empty values (records on another branch arm) say nothing; the count at the end covers them.
  const values = c.nulls ? c.preview.filter((x) => x !== null) : c.preview;
  const shown = values.map((x) => formatValue(x, name)).join(", ");
  return `${shown}${c.rows > c.preview.length ? ", …" : ""}${c.nulls ? `  (${c.nulls} of ${c.rows} empty)` : ""}`;
}
