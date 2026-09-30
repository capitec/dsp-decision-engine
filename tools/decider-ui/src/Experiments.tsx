import { formatValue, recordLabel, type RecordKey } from "./model/protocol";
import { experimentHeadline, isRunnableScenario, statusLabel, type ExperimentResult, type ScenarioResult } from "./model/experiment";
import { SankeyDiagram } from "./Sankey";

interface Props {
  result: ExperimentResult;
  /** Launch the equivalent debugger run for a scenario (and, when given, one record of it). */
  onDebug: (scenario: ScenarioResult, row?: number) => void;
  keyCol?: RecordKey;
}

/** The experiments result: run state, per-scenario outcomes, aggregate summary and a path Sankey. */
export function Experiments({ result, onDebug, keyCol }: Props) {
  const scenarios = result.scenarios ?? [];
  const summaryByName = new Map((result.summary?.scenarios ?? []).map((s) => [s.name, s]));
  return (
    <div className="experiments">
      <div className="experiment-head">
        <span className={`status badge-${result.status}`}>{statusLabel(result.status)}</span>
        <strong>{result.name || result.manifest?.revision?.authored || "experiment"}</strong>
        <span className="muted">{experimentHeadline(result)}</span>
      </div>

      {result.status === "non_reproducible" && <Banner kind="warn">A baseline rerun diverged: this run is not a deterministic reproduction, so its comparisons are flagged, not trusted.</Banner>}
      {result.status === "partial" && <Banner kind="warn">Some scenarios did not finish; the ones below that failed are shown with their error, not a comparison.</Banner>}
      {result.status === "cancelled" && <Banner>Run cancelled; completed scenarios are shown.</Banner>}
      {result.status === "timed_out" && <Banner>Run timed out; completed scenarios are shown.</Banner>}

      {result.summary?.first_divergence && (
        <div className="muted first-divergence">
          First divergence at <span className="mono">{result.summary.first_divergence.location}</span> in <strong>{result.summary.first_divergence.scenario}</strong>
        </div>
      )}

      <table className="scenarios-table">
        <thead>
          <tr>
            <th>scenario</th>
            <th>state</th>
            <th>divergences</th>
            <th>changed values</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {scenarios.map((s) => {
            const summary = summaryByName.get(s.name);
            const changed = Object.values(summary?.changed ?? {}).reduce((t, rows) => t + rows.length, 0);
            return (
              <tr key={s.name} className={s.status}>
                <td className="mono">{s.name}</td>
                <td>{s.status}</td>
                <td className="mono">{s.divergences?.length ?? 0}</td>
                <td className="mono">{changed}</td>
                <td>
                  {isRunnableScenario(s) && <a className="link" onClick={() => onDebug(s)}>debug</a>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {result.summary && result.summary.scenarios.some((s) => Object.keys(s.drill_down).length) && (
        <DrillDown result={result} onDebug={onDebug} keyCol={keyCol} />
      )}

      {result.sankey && <SankeyDiagram sankey={result.sankey} />}

      {scenarios.filter((s) => s.error).map((s) => (
        <div key={s.name} className="scenario-error">
          <strong>{s.name}</strong>: <span className="mono">{s.error}</span>
        </div>
      ))}
    </div>
  );
}

function DrillDown({ result, onDebug, keyCol }: { result: ExperimentResult; onDebug: Props["onDebug"]; keyCol?: RecordKey }) {
  const rows = (result.summary?.scenarios ?? []).flatMap((s) =>
    Object.entries(s.drill_down).flatMap(([col, samples]) =>
      samples.map((d) => ({ scenario: s.name, col, ...d })),
    ),
  );
  if (!rows.length) return null;
  return (
    <details className="drill-down" open>
      <summary>Changed records (sampled)</summary>
      <table>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.scenario}-${r.col}-${r.row}`}>
              <td className="mono">{r.scenario}</td>
              <td className="mono">{r.col}</td>
              <td>{keyCol ? recordLabel(r.row, keyCol) : `row ${r.row}`}</td>
              <td className="mono"><s>{formatValue(r.expected, r.col)}</s> {formatValue(r.actual, r.col)}</td>
              <td>
                <a className="link" onClick={() => { const s = result.scenarios.find((x) => x.name === r.scenario); if (s) onDebug(s, r.row); }}>debug this record</a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

function Banner({ children, kind }: { children: React.ReactNode; kind?: "warn" }) {
  return <div className={`experiment-banner ${kind ?? ""}`}>{children}</div>;
}
