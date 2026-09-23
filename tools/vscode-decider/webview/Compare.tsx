import { useState } from "react";
import type { Comparison, ValueDiff } from "../src/compare";
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

/** Two runs side by side, step by step, in execution order. */
export function Compare({ comparison: c, busy, error, record, onSelect, onCompareRevision, onOpenDiff }: Props) {
  const [onlyChanges, setOnlyChanges] = useState(true);
  const toolbar = (
    <div className="actions">
      <button onClick={onCompareRevision}>Compare with a git revision…</button>
    </div>
  );
  if (busy) return <div className="compare"><div className="empty">{busy}</div></div>;
  if (error) return <div className="compare"><div className="empty error">{error}</div>{toolbar}</div>;
  if (!c)
    return (
      <div className="compare">
        <div className="empty">Nothing compared yet. Try a what-if on the Params tab, run scenarios, or compare with a git revision.</div>
        {toolbar}
      </div>
    );
  const steps = c.steps.filter((s) => {
    if (!onlyChanges) return true;
    if (s.status === "same" || s.status === "not run") return false;
    return record === null || s.status !== "changed" || s.structural.length > 0 || s.outputs.some((o) => o.changedRows.includes(record));
  });
  const changedOut = new Set(c.output.map((o) => o.name));
  const count = (st: string) => c.steps.filter((s) => s.status === st).length;
  return (
    <div className="compare">
      <div className="summary">
        <span><span className="muted">Baseline</span> <strong>{c.a}</strong> <span className="muted">vs</span> <strong>{c.b}</strong></span>
        <label className="right"><input type="checkbox" checked={onlyChanges} onChange={(e) => setOnlyChanges(e.target.checked)} /> only changes</label>
      </div>
      <div className="muted">{c.rows} records · {count("changed")} steps changed, {count("added")} added, {count("removed")} removed</div>
      {(c.errors.a || c.errors.b) && (
        <div className="error">
          {c.errors.a && <div>{c.a}: {c.errors.a}</div>}
          {c.errors.b && <div>{c.b}: {c.errors.b}</div>}
        </div>
      )}
      <h4>Final outputs</h4>
      <Diffs diffs={c.output} record={record} keyCol={c.key} />
      <div className="unchanged-list">
        {c.outputColumns.filter((o) => !changedOut.has(o)).map((o) => (
          <span key={o} className="chip small same">{o}: unchanged</span>
        ))}
      </div>
      <h4>
        Step by step
        {c.firstDivergence && (
          <span className="muted small"> · first difference at <a onClick={() => onSelect(c.firstDivergence!)}>{c.firstDivergence}</a></span>
        )}
      </h4>
      {steps.length === 0 && <div className="muted">Every step produces the same values.</div>}
      {steps.map((s) => (
        <div key={s.path} className={`step-diff ${s.status}`}>
          <div>
            <span className={`badge-status ${s.status}`}>{s.status}</span> <a onClick={() => onSelect(s.path)}>{s.path}</a>
            {s.structural.map((x) =>
              x === "code" && c.files ? (
                <a key={x} className="chip small" onClick={() => onOpenDiff(s.path)} title="Open a diff of the two versions">view code diff</a>
              ) : (
                <span key={x} className="chip small">{x} changed</span>
              ),
            )}
          </div>
          {s.outputs.length > 0 && <Diffs diffs={s.outputs} record={record} keyCol={c.key} />}
        </div>
      ))}
      {toolbar}
    </div>
  );
}
