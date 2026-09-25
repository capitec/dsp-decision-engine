import type { Comparison } from "./compare";
import type { Scenario, Sweep } from "./sweep";

// Shapes the Python bridge sends, and the messages between a host and the UI.

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
  python: { file: string; line: number; bodyLine: number | null; endLine?: number | null } | null;
  /** Changes when the code or config behind the node changes. */
  code: string;
  /** The first line of the step's docstring. */
  doc?: string;
  /** What a one-line step returns, e.g. `min(pl_raw_rate, repo_rate + cap_margin)`. */
  formula?: string | null;
  /** The step's source, when short. */
  body?: string | null;
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
  /** What the flow decides: its emitted values and what its top-level branches set. */
  outcome?: string[];
  /** The file `PARAMS` is read from, when the module names one. */
  valuesFile?: string | null;
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
  /** Inside a loop: which iteration. */
  iteration?: number;
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
  /** Set when a value or iteration breakpoint paused the run. */
  hit?: Hit | null;
  /** Every step and group that has run since the start or the last rewind. */
  ran?: string[];
}

/** Send a branch's records down `arm`, or run a loop exactly `iterations` times; for one record, or all. */
export interface Force {
  path: string;
  arm?: number;
  iterations?: number;
  row?: number | null;
}

/** Pause when `name op value` first holds for a record, just after a step inside `scope` writes it; or before a loop's `iteration`. */
/** A breakpoint: `{path}` pauses before that step; `{path, iteration}` before a loop's iteration; `{name, op, value}` when a value meets a condition. */
export interface Watch {
  name?: string;
  op?: "==" | "!=" | "<" | "<=" | ">" | ">=";
  value?: unknown;
  scope?: string[];
  row?: number | null;
  path?: string;
  iteration?: number;
}

export interface Controls {
  forces: Force[];
  watches: Watch[];
}

/** Why a watch paused the run: which records newly meet it there, and their values. */
export interface Hit {
  watch: number;
  text: string;
  path?: string;
  rows?: number[];
  values?: unknown[];
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
  /** Records a comparison forced; the others ran as they are. */
  forcedRows?: number[];
  /** Set when the value came from you: "force@<branch or loop>" or "override@<path>", and what it was before. */
  setBy?: string;
  was?: unknown;
  /** Not set yet for the focused record: there is nothing to break down. */
  unset?: boolean;
  /** For a forced branch condition: its arms by name, so a value reads as the arm it picks. */
  arms?: string[];
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
  hit?: Hit | null;
}

/** One change to a value: the step (or "override@<path>" for a value you set) and iteration that made it. */
export interface ValueChange {
  /** Its index in the run's timeline, for going back to it. */
  change: number;
  path: string;
  iteration: number | null;
  arm: number | null;
  /** For the focused record. */
  value?: unknown;
  before?: unknown;
  /** The step wrote the value the record already had. */
  kept?: boolean;
  /** Undone by going back: it happens again as the run goes on. */
  pending?: boolean;
  /** The step copied this input into the value unchanged, and the step that computed that input (null: the data). */
  via?: string;
  viaPath?: string | null;
  /** For the whole batch: how many records changed, and the first few new values. */
  rows?: number;
  values?: unknown[];
}

export interface ValueHistory {
  name: string;
  /** Whether it is an input column, and its value there. */
  input: boolean;
  initial: unknown;
  changes: ValueChange[];
}

/** The column that names a record, and its value on every row. */
export type RecordKey = { name: string; values: unknown[] } | null;

/** "cap_by_income skipped" or "cap_by_income edited", for one entry of `RunStatus.edits`. */
export const editLabel = ([path, action]: [string, string]) => `${lastSegment(path)} ${action === "delete" ? "skipped" : "edited"}`;

/** "client_id 2", or "row 1" when the data has no id column. */
export function recordLabel(row: number, key: RecordKey): string {
  return key ? `${key.name} ${String(key.values[row])}` : `row ${row}`;
}

/** Names whose values read as percentages: rates, loadings, discounts, margins. */
export const isRateName = (name?: string) => !!name && /(rate|loading|discount|margin|share)s?$/.test(name);
/** Names whose values are rand amounts. */
const isMoneyName = (name?: string) => !!name && /(amount|cost|income|fee|instalment|expenses|offer)s?$/.test(name);

/**
 * A value as the UI shows it: no float noise, no thousands separators (4000, not 4,000); amounts of 100 or more to two decimals (58113.07), smaller ones
 * to four. With its column's `name`, a rate below 1 shows as a percentage ("25.2%").
 */
export function formatValue(v: unknown, name?: string): string {
  if (v === undefined) return "—";
  if (v === null) return "empty";
  if (typeof v === "number" && isRateName(name) && Math.abs(v) < 1) return `${Number((v * 100).toFixed(2))}%`;
  if (typeof v === "number" && (isMoneyName(name) || (!!name && /cap$/.test(name) && Math.abs(v) >= 1000))) return `R\u00a0${v.toLocaleString("en-US", { useGrouping: false, minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  if (typeof v === "number") return v.toLocaleString("en-US", { useGrouping: false, maximumFractionDigits: Math.abs(v) >= 100 ? 2 : 4 });
  return typeof v === "string" ? v : JSON.stringify(v);
}

export type Tab = "graph" | "state" | "params" | "scenarios" | "compare";

/** Messages from the host to the UI. */
export type ToUI =
  | { type: "describe"; describe: DescribeResult }
  | ({ type: "status" } & RunStatus)
  | { type: "state"; columns: ColumnSummary[] | null; rows: number; key: RecordKey }
  | { type: "lineage"; lineage: Lineage | null; history: ValueHistory | null }
  | { type: "treePath"; path: string; row: number; visited: string[]; result?: unknown[] }
  | { type: "compare"; comparison: Comparison | null; busy?: string; error?: string }
  | { type: "tab"; tab: Tab }
  | { type: "select"; path: string }
  | { type: "edited"; path: string; formula: string | null; restored?: boolean }
  | { type: "sweep"; sweep: Sweep | null; busy?: string; error?: string };

/** Messages from the UI to the host. */
export type FromUI =
  | { type: "ready" }
  | { type: "reveal"; path: string }
  | { type: "record"; row: number | null }
  | { type: "lineage"; name: string }
  | { type: "treePath"; path: string }
  | { type: "rewind"; path: string }
  /** A change's index in the timeline, or a step's path: just after it last wrote. */
  | { type: "goTo"; change: number | string }
  /** Go back to just before a forced branch or loop's condition, then run on to `back` (where the run was paused). */
  | { type: "rerun"; path: string; back?: string; when?: "before" | "after" }
  | { type: "skip"; path: string }
  | { type: "reloadStep"; path: string }
  | { type: "restore"; path: string }
  | { type: "compareEdits"; label: string; edits: Record<string, "delete" | "replace">; path?: string }
  | { type: "whatIf"; params: unknown; overrides: Record<string, unknown>; row: number | null; label: string; forces?: Force[] }
  | { type: "setControls"; controls: Controls }
  | { type: "compareForces"; a: { label: string; forces: Force[] }; b: { label: string; forces: Force[] } }
  | { type: "restartWith"; params: unknown }
  | { type: "compareRevision" }
  /** Run the flow from the start, pausing at its breakpoints. */
  | { type: "run" }
  | { type: "runTo"; path: string }
  | { type: "maximise" }
  | { type: "step" }
  /** Attach the Python debugger and stop on the first line of the step the run is paused before. */
  | { type: "debugStep" }
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
