import { useState } from "react";
import { same, type Comparison, type ValueDiff } from "../src/compare";
import { formatValue, recordLabel, type RecordKey } from "../src/protocol";

interface Props {
  comparison: Comparison | null;
  busy?: string;
  error?: string;
  record: number | null;
  onSelect: (path: string) => void;
  onCompareRevision: () => void;
  onOpenDiff: (path: string) => void;
}

function Diffs({ diffs, record, keyCol }: { diffs: ValueDiff[]; record: number | null; keyCol: RecordKey }) {
  return (
    <table className="vdiffs">
      <tbody>
        {diffs.flatMap((d) =>
          d.samples.map((s, i) => (
            <tr key={`${d.name}-${s.row}`} className={record === s.row ? "hit" : ""}>
              <td className="mono">{i === 0 ? d.name : ""}</td>
              <td>{recordLabel(s.row, keyCol)}</td>
              <td className="mono before">{formatValue(s.a)}</td>
              <td className="arrow">→</td>
              <td className="mono after">{formatValue(s.b)}</td>
              <td className="muted">{i === 0 && d.changedRows.length > d.samples.length ? `+${d.changedRows.length - d.samples.length} more` : ""}</td>
            </tr>
          )),
        )}
      </tbody>
    </table>
  );
}

/** Each result column, changed or not, with its values: the answer to "did the decision move?". */
function Results({ c, record }: { c: Comparison; record: number | null }) {
  const rows = record === null ? Array.from({ length: c.rows }, (_, i) => i) : [record];
  const names = Object.keys(c.results.b);
  return (
    <table className="results">
      <thead>
        <tr>
          <th />
          {rows.map((r) => (
            <th key={r}>{recordLabel(r, c.key)}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {names.map((n) => (
          <tr key={n}>
            <td className="mono">{n}</td>
            {rows.map((r) => {
              const a = c.results.a[n]?.[r];
              const b = c.results.b[n]?.[r];
              return same(a, b) ? (
                <td key={r} className="mono unchanged-cell">{formatValue(b)} <span className="muted small">same</span></td>
              ) : (
                <td key={r} className="mono changed-cell"><span className="before">{formatValue(a)}</span> → <span className="after">{formatValue(b)}</span></td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Two runs side by side, step by step, in execution order. */
export function Compare({ comparison: c, busy, error, record, onSelect, onCompareRevision, onOpenDiff }: Props) {
  const [onlyChanges, setOnlyChanges] = useState(true);
  const revisionButton = <button onClick={onCompareRevision}>Compare with a git revision…</button>;
  if (busy) return <div className="compare"><div className="empty">{busy}</div></div>;
  if (error) return <div className="compare"><div className="empty error">{error}</div><div className="actions">{revisionButton}</div></div>;
  if (!c)
    return (
      <div className="compare">
        <div className="empty">Nothing compared yet. Try a what-if, run scenarios, or compare with a git revision.</div>
        <div className="actions">{revisionButton}</div>
      </div>
    );
  const steps = c.steps.filter((s) => {
    if (!onlyChanges) return true;
    if (s.status === "same" || s.status === "not run") return false;
    return record === null || s.status !== "changed" || s.structural.length > 0 || s.outputs.some((o) => o.changedRows.includes(record));
  });
  const count = (st: string) => c.steps.filter((s) => s.status === st).length;
  return (
    <div className="compare">
      <div className="summary">
        <span><span className="muted">Baseline</span> <strong>{c.a}</strong> <span className="muted">vs</span> <strong>{c.b}</strong></span>
        <span className="right">{revisionButton}</span>
      </div>
      <div className="muted">{c.rows} records · {count("changed")} steps changed{count("added") ? `, ${count("added")} added` : ""}{count("removed") ? `, ${count("removed")} removed` : ""}</div>
      {(c.errors.a || c.errors.b) && (
        <div className="error">
          {c.errors.a && <div>{c.a}: {c.errors.a}</div>}
          {c.errors.b && <div>{c.b}: {c.errors.b}</div>}
        </div>
      )}
      {c.changedInputs.length > 0 && (
        <>
          <h4>Changed inputs</h4>
          {c.changedInputs.map((i) => (
            <div key={i.name} className="mono">{i.name} → {formatValue(i.after)} <span className="muted">for {i.scope}</span></div>
          ))}
        </>
      )}
      <h4>Results</h4>
      <Results c={c} record={record} />
      <h4>
        Step by step
        {c.firstDivergence && (
          <span className="muted small"> · first difference at <a onClick={() => onSelect(c.firstDivergence!)}>{c.firstDivergence}</a></span>
        )}
        <label className="right small"><input type="checkbox" checked={onlyChanges} onChange={(e) => setOnlyChanges(e.target.checked)} /> only changed steps</label>
      </h4>
      {steps.length === 0 && <div className="muted">Every step produces the same values.</div>}
      {steps.map((s) => (
        <div key={s.path} className={`step-diff ${s.status}`}>
          <div>
            <span className={`badge-status ${s.status}`}>{s.status}</span> <a onClick={() => onSelect(s.path)}>{s.path}</a>
            {s.paramChanges.map((p) => (
              <span key={p} className="chip small">{p}</span>
            ))}
            {s.structural.includes("code") &&
              (c.files ? (
                <a className="chip small" onClick={() => onOpenDiff(s.path)} title="Open a diff of the two versions">code changed · view diff</a>
              ) : (
                <span className="chip small">code changed</span>
              ))}
            {s.structural.filter((x) => x === "reads" || x === "writes").map((x) => (
              <span key={x} className="chip small">{x} changed</span>
            ))}
          </div>
          {s.outputs.length > 0 && <Diffs diffs={s.outputs} record={record} keyCol={c.key} />}
        </div>
      ))}
    </div>
  );
}
