import { scenarioDefSummary, typedFields, type ExperimentDef, type MetricDef, type PlotDef, type ReportBlock } from "./model/experiment";

interface Props {
  definition: ExperimentDef;
  /** Whether a `result.json` sits next to this experiment already. */
  hasResult: boolean;
  /** Every problem `experiment_validate` found. */
  problems: string[];
  onRun: () => void;
  onRunOnDataset: () => void;
  onOpenReport: () => void;
  onClose: () => void;
}

/** A read-only summary of `experiment.json`: its scenarios, metrics, plots and report, with run/open actions.
 * Editing stays in Python (edit the file, or build it with the `Experiment` API) — this view is for browsing
 * and running what's already there. */
export function ExperimentSummary({ definition, hasResult, problems, onRun, onRunOnDataset, onOpenReport, onClose }: Props) {
  return (
    <div className="experiment-summary">
      <header>
        <strong className="pipeline">{definition.name || "experiment"}</strong>
        <button className="primary" onClick={onRun}>▶ Run</button>
        <button onClick={onRunOnDataset}>Run on dataset…</button>
        {hasResult && <button onClick={onOpenReport}>Open last report</button>}
        <button className="link" style={{ marginLeft: "auto" }} onClick={onClose}>back to flow</button>
      </header>

      {problems.length > 0 && (
        <ul className="problems">
          {problems.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
      )}

      <div className="summary-body">
        {definition.description && <p className="description">{definition.description}</p>}

        <dl className="meta-grid">
          <dt>flow</dt>
          <dd className="mono">{definition.flow.entry}</dd>
          <dt>input</dt>
          <dd className="mono">{definition.input?.path || "(none set)"}</dd>
          <dt>revision</dt>
          <dd className="mono">{definition.revision || "HEAD"}</dd>
        </dl>

        <Section title={`Scenarios (${definition.scenarios.length})`}>
          {definition.scenarios.length === 0 && <p className="empty">no scenarios; only the baseline would run</p>}
          {definition.scenarios.map((s) => (
            <div key={s.name} className="summary-row">
              <span className="mono name">{s.name}</span>
              <span className="muted">{scenarioDefSummary(s)}</span>
            </div>
          ))}
        </Section>

        <Section title={`Metrics (${definition.metrics.length})`}>
          {definition.metrics.map((m, i) => (
            <MetricOrPlotRow key={i} t={m} skip={METRIC_OWN} />
          ))}
        </Section>

        <Section title={`Plots (${definition.plots.length})`}>
          {definition.plots.map((p, i) => (
            <MetricOrPlotRow key={i} t={p} skip={PLOT_OWN} />
          ))}
        </Section>

        <Section title="Report">
          {definition.report ? <BlockOutline blocks={definition.report.blocks} /> : <p className="muted">default layout: an overview, then a section per metric</p>}
        </Section>
      </div>
    </div>
  );
}

const METRIC_OWN = new Set(["type", "name", "nodes", "node_id"]);
const PLOT_OWN = new Set(["type", "metric", "name"]);

function MetricOrPlotRow({ t, skip }: { t: MetricDef | PlotDef; skip: Set<string> }) {
  const extra = typedFields(t, skip);
  return (
    <div className="summary-row">
      <span className="mono name">{t.name || t.type}</span>
      <span className="muted">
        {t.type}
        {"nodes" in t && t.nodes ? ` at ${String(t.nodes)}` : ""}
        {"node_id" in t && t.node_id ? ` at step ${String(t.node_id)}` : ""}
        {"metric" in t ? ` draws ${String((t as PlotDef).metric)}` : ""}
        {extra.length > 0 && ` (${extra.map(([k, v]) => `${k}=${v}`).join(", ")})`}
      </span>
    </div>
  );
}

function BlockOutline({ blocks, depth = 0 }: { blocks: ReportBlock[]; depth?: number }) {
  return (
    <div className="block-outline" style={{ paddingLeft: depth ? 14 : 0 }}>
      {blocks.map((b, i) => (
        <div key={i}>
          <div className="summary-row">
            <span className="mono name">{b.type}</span>
            <span className="muted">
              {"metric" in b ? String(b.metric) : "plot" in b ? String(b.plot) : "title" in b ? String(b.title) : ""}
            </span>
          </div>
          {Array.isArray(b.blocks) && <BlockOutline blocks={b.blocks} depth={depth + 1} />}
        </div>
      ))}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="summary-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}
