import {
  experimentHeadline,
  plotsForMetric,
  renderMarkdownLite,
  reportOf,
  scenarioChanges,
  statusLabel,
  summaryTable,
  type ExperimentResult,
  type ReportBlock,
  type ScenarioResult,
} from "./model/experiment";
import { VegaChart } from "./VegaChart";

interface Props {
  result: ExperimentResult;
  /** Launch the equivalent debugger run for a scenario. */
  onDebug: (scenario: ScenarioResult) => void;
}

/** An experiment's result: its report, walked block by block (the run's own, or Python's default). */
export function Experiments({ result, onDebug }: Props) {
  const report = reportOf(result);
  return (
    <div className="experiments">
      <div className="experiment-head">
        <span className={`status badge-${result.status}`}>{statusLabel(result.status)}</span>
        <strong>{result.name || result.manifest?.revision?.authored || "experiment"}</strong>
        <span className="muted">{experimentHeadline(result)}</span>
      </div>

      {result.status === "non_reproducible" && <Banner kind="warn">A baseline rerun diverged: this run is not a deterministic reproduction, so its comparisons are flagged, not trusted.</Banner>}
      {result.status === "partial" && <Banner kind="warn">Some scenarios did not finish; the ones below that failed are shown with their error.</Banner>}
      {result.status === "cancelled" && <Banner>Run cancelled; completed scenarios are shown.</Banner>}
      {result.status === "timed_out" && <Banner>Run timed out; completed scenarios are shown.</Banner>}

      {report.title && <h1>{report.title}</h1>}
      {report.blocks.map((b, i) => (
        <Block key={i} block={b} result={result} onDebug={onDebug} />
      ))}
    </div>
  );
}

function Block({ block, result, onDebug }: { block: ReportBlock; result: ExperimentResult; onDebug: Props["onDebug"] }) {
  switch (block.type) {
    case "overview":
      return <Overview result={result} onDebug={onDebug} />;
    case "text":
      return <p className="report-text" dangerouslySetInnerHTML={{ __html: renderMarkdownLite(String(block.markdown ?? "")) }} />;
    case "table":
      return <SummaryTable metric={String(block.metric)} result={result} />;
    case "chart": {
      const figure = result.plots[String(block.plot)];
      return figure ? <VegaChart figure={figure} /> : <p className="empty">no plot named {String(block.plot)}</p>;
    }
    case "metric_section":
      return <MetricSection metric={String(block.metric)} result={result} />;
    case "row":
      return (
        <div className="report-row">
          {(block.blocks ?? []).map((b, i) => (
            <Block key={i} block={b} result={result} onDebug={onDebug} />
          ))}
        </div>
      );
    case "section":
      return (
        <section className="report-section">
          <h2>{String(block.title ?? "")}</h2>
          {(block.blocks ?? []).map((b, i) => (
            <Block key={i} block={b} result={result} onDebug={onDebug} />
          ))}
        </section>
      );
    default:
      return null;
  }
}

function Overview({ result, onDebug }: { result: ExperimentResult; onDebug: Props["onDebug"] }) {
  return (
    <table className="scenarios-table">
      <thead>
        <tr>
          <th>scenario</th>
          <th>state</th>
          <th>records</th>
          <th>changes</th>
          <th>error</th>
          <th />
        </tr>
      </thead>
      <tbody>
        {result.scenarios.map((s) => (
          <tr key={s.name} className={s.status}>
            <td className="mono">{s.name}</td>
            <td>{s.status}</td>
            <td className="mono">{s.records}</td>
            <td className="muted">{scenarioChanges(s)}</td>
            <td className="mono">{s.error ?? ""}</td>
            <td>
              <a className="link" onClick={() => onDebug(s)}>
                debug
              </a>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SummaryTable({ metric, result }: { metric: string; result: ExperimentResult }) {
  const { columns, rows } = summaryTable(result.summaries[metric] ?? {});
  if (!rows.length) return <p className="empty">nothing to show</p>;
  return (
    <div className="table">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c} className="mono">
                  {r[c] ?? ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricSection({ metric, result }: { metric: string; result: ExperimentResult }) {
  const plots = plotsForMetric(result.definition, metric);
  return (
    <section className="metric-section">
      <h3>{metric}</h3>
      <SummaryTable metric={metric} result={result} />
      <div className="report-row">
        {plots.map((key) => result.plots[key] && <VegaChart key={key} figure={result.plots[key]} />)}
      </div>
    </section>
  );
}

function Banner({ children, kind }: { children: React.ReactNode; kind?: "warn" }) {
  return <div className={`experiment-banner ${kind ?? ""}`}>{children}</div>;
}
