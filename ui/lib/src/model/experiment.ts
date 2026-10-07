// `experiment.json` (`ExperimentDef`) and its result as `decider.lab.experiments.run` serialises them, plus the pure
// helpers the summary and results views render with. Kept JSON-shaped on purpose: the wire form is the contract.

export type ExperimentRunStatus = "completed" | "non_reproducible" | "partial" | "failed" | "cancelled" | "timed_out";
export type ScenarioStatus = "completed" | "failed";

export interface FlowSpec {
  entry: string;
  decider?: string;
}

export interface InputSpec {
  path?: string;
}

export interface ScenarioDef {
  name: string;
  revision?: string | null;
  params?: Record<string, unknown>;
  overrides?: Record<string, unknown>;
  at?: string | null;
  row?: number | null;
}

/** A registered `Metric`/`Plot`/report `Block`: `type` picks the shape, the rest is type-specific and kept open
 * so a definition round-trips through editing even for a type this build doesn't know about. */
export interface Typed {
  type: string;
  [key: string]: unknown;
}

export interface MetricDef extends Typed {
  name?: string;
  nodes?: string | null;
  node_id?: string | null;
}

export interface PlotDef extends Typed {
  metric: string;
  name?: string;
  width?: number | null;
  height?: number | null;
}

export type ReportBlock = Typed & { blocks?: ReportBlock[] };

export interface Report {
  title?: string;
  blocks: ReportBlock[];
}

export interface ExperimentDef {
  version?: string;
  name: string;
  description?: string;
  flow: FlowSpec;
  revision?: string;
  input?: InputSpec;
  scenarios: ScenarioDef[];
  metrics: MetricDef[];
  plots: PlotDef[];
  batch_size?: number;
  determinism?: boolean;
  report?: Report | null;
}

/** A plot or chart, already rendered: `vega-embed` draws either grammar from the same `spec`. */
export interface Figure {
  grammar: "vega-lite" | "vega";
  spec: Record<string, unknown>;
}

/** One metric's result for one scenario; its shape depends on the metric type, so it stays open. */
export type MetricSummary = Record<string, unknown>;

export interface ScenarioResult {
  name: string;
  status: ScenarioStatus;
  error?: string | null;
  revision?: { authored: string; resolved?: string | null } | null;
  params?: Record<string, unknown>;
  overrides?: Record<string, unknown>;
  at?: string | null;
  row?: number | null;
  records: number;
  output_ref?: { name: string; location: string } | null;
}

export interface ExperimentResult {
  manifest?: {
    revision?: { authored?: string; resolved?: string | null } | null;
    input?: { dataset: string; fingerprint: string | null; columns: [string, string][]; row_count: number | null } | null;
  };
  status: ExperimentRunStatus;
  name?: string;
  nondeterministic?: boolean;
  scenarios: ScenarioResult[];
  job?: { message?: string; error?: string | null; done?: number; total?: number | null } | null;
  summaries: Record<string, Record<string, MetricSummary>>;
  plots: Record<string, Figure>;
  definition?: ExperimentDef | null;
}

/** One registered type's JSON Schema, from the `experiment_schema` bridge command. */
export interface TypeSchema {
  properties: Record<string, { type?: string; default?: unknown; description?: string; [key: string]: unknown }>;
  required?: string[];
  /** The Python `Summary` subclass's name: what a metric gives, or what a plot needs; matching the two picks valid plots for a metric. */
  summary?: string;
  requires?: string;
  [key: string]: unknown;
}

export interface ExperimentSchema {
  metrics: Record<string, TypeSchema>;
  plots: Record<string, TypeSchema>;
  blocks: Record<string, TypeSchema>;
}

/** The plot types whose `requires` matches the metric's `summary`, for the plot type picker. */
export function plotTypesFor(schema: ExperimentSchema, metricType: string): string[] {
  const summary = schema.metrics[metricType]?.summary;
  return Object.entries(schema.plots)
    .filter(([, s]) => s.requires === summary)
    .map(([type]) => type);
}

/** One line that says what the run is, never letting a non-reproducible or partial run read as done. */
export function experimentHeadline(r: ExperimentResult): string {
  const scenarioCount = r.scenarios.length;
  const bits = [`${scenarioCount} scenario${scenarioCount === 1 ? "" : "s"}`];
  if (r.status === "non_reproducible") bits.push("non-reproducible: a baseline rerun diverged");
  if (r.status === "partial") bits.push("partial: some scenarios did not finish");
  if (r.status === "cancelled") bits.push("cancelled");
  if (r.status === "timed_out") bits.push("timed out");
  return bits.join(" · ");
}

/** The status as the badge reads: the five non-completed states are named, never "done". */
export function statusLabel(status: ExperimentRunStatus): string {
  return (
    {
      completed: "completed",
      non_reproducible: "non-reproducible",
      partial: "partial",
      failed: "failed",
      cancelled: "cancelled",
      timed_out: "timed out",
    }[status] ?? status
  );
}

/** The report to render: the run's own, or Python's default (an overview, then a section per metric). */
export function reportOf(r: ExperimentResult): Report {
  return r.definition?.report ?? { title: "", blocks: defaultReportBlocks(r) };
}

function defaultReportBlocks(r: ExperimentResult): ReportBlock[] {
  return [{ type: "overview" }, ...Object.keys(r.summaries).map((metric) => ({ type: "metric_section", metric }))];
}

/** The plots that draw `metric`, by their key in `result.plots` (a plot's own `name`, or its `metric`). */
export function plotsForMetric(def: ExperimentDef | null | undefined, metric: string): string[] {
  return (def?.plots ?? []).filter((p) => p.metric === metric).map((p) => (p.name ? p.name : p.metric));
}

/** One summary flattened into a table row's cells, nested objects joined with `.` like the Python report. */
export function flattenSummary(summary: MetricSummary): Record<string, string> {
  const out: Record<string, string> = {};
  const walk = (value: unknown, prefix: string) => {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      for (const [k, v] of Object.entries(value)) walk(v, prefix ? `${prefix}.${k}` : k);
    } else {
      out[prefix] = formatCell(value);
    }
  };
  for (const [k, v] of Object.entries(summary)) walk(v, k);
  return out;
}

function formatCell(value: unknown): string {
  if (value == null) return "";
  if (Array.isArray(value)) {
    const shown = value.slice(0, 6).map((v) => (typeof v === "number" ? String(Math.round(v * 10000) / 10000) : String(v)));
    return `[${shown.join(", ")}${value.length > 6 ? ", …" : ""}] (${value.length})`;
  }
  return String(value);
}

/** A table's rows (one per scenario) and the union of columns across them, for a generic summary table. */
export function summaryTable(byScenario: Record<string, MetricSummary>): { columns: string[]; rows: Record<string, string>[] } {
  const rows = Object.entries(byScenario).map(([scenario, s]) => ({ scenario, ...flattenSummary(s) }));
  const columns = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  return { columns, rows };
}

/** What a scenario changed, for the overview table: its params, overrides (and where), or another revision. */
export function scenarioChanges(s: ScenarioResult): string {
  const bits: string[] = [];
  if (s.params && Object.keys(s.params).length) bits.push(`params ${JSON.stringify(s.params)}`);
  if (s.overrides && Object.keys(s.overrides).length) {
    const where = (s.at ? ` after ${s.at}` : "") + (s.row != null ? ` on record ${s.row}` : "");
    bits.push(`set ${JSON.stringify(s.overrides)}${where}`);
  }
  if (s.revision) bits.push(`revision ${s.revision.authored}`);
  return bits.join("; ") || "unchanged";
}

/** What a scenario (in the definition, not a result) changes, for the read-only summary. */
export function scenarioDefSummary(s: ScenarioDef): string {
  const bits: string[] = [];
  if (s.params && Object.keys(s.params).length) bits.push(`params ${JSON.stringify(s.params)}`);
  if (s.overrides && Object.keys(s.overrides).length) {
    const where = (s.at ? ` after ${s.at}` : "") + (s.row != null ? ` on record ${s.row}` : "");
    bits.push(`set ${JSON.stringify(s.overrides)}${where}`);
  }
  if (s.revision) bits.push(`revision ${s.revision}`);
  return bits.join("; ") || "unchanged";
}

/** A `Typed`'s fields beyond the given ones, as `key: value` pairs, for a read-only metric/plot/block line. */
export function typedFields(t: Typed, skip: Set<string>): [string, string][] {
  return Object.entries(t)
    .filter(([k, v]) => !skip.has(k) && v != null && v !== "")
    .map(([k, v]) => [k, typeof v === "object" ? JSON.stringify(v) : String(v)]);
}

/** `markdown` as HTML: escaped first, then `**bold**` and `*italic*`, paragraphs on blank lines.
 * Not CommonMark — a `text` block's prose rarely needs more, and a full parser is a dependency for 20 lines. */
export function renderMarkdownLite(markdown: string): string {
  const escape = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const inline = (s: string) =>
    escape(s)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(?<!\*)\*(?!\*)(.+?)\*(?!\*)/g, "<em>$1</em>")
      .replace(/\n/g, "<br/>");
  return markdown
    .split(/\n\s*\n/)
    .map((p) => `<p>${inline(p)}</p>`)
    .join("");
}
