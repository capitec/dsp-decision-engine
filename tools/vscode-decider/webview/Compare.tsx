import { useState } from "react";
import type { Comparison, ValueDiff } from "../src/compare";

interface Props {
  comparison: Comparison | null;
  busy?: string;
  error?: string;
  record: number | null;
  onSelect: (path: string) => void;
}

const fmt = (v: unknown) => (v === undefined ? "—" : JSON.stringify(v));

function Diffs({ diffs, record }: { diffs: ValueDiff[]; record: number | null }) {
  return (
    <>
      {diffs.map((d) => {
        const hit = record !== null && d.changedRows.includes(record);
        return (
          <div key={d.name} className={`vdiff ${hit ? "hit" : ""}`}>
            <span className="mono">{d.name}</span>
            <span className="muted"> · {d.changedRows.length} row{d.changedRows.length === 1 ? "" : "s"}</span>
            {d.samples.map((s) => (
              <span key={s.row} className="mono sample">
                {" "}r{s.row}: <del>{fmt(s.a)}</del> → <ins>{fmt(s.b)}</ins>
              </span>
            ))}
          </div>
        );
      })}
    </>
  );
}

/** Two runs side by side, step by step, in execution order. */
export function Compare({ comparison: c, busy, error, record, onSelect }: Props) {
  const [onlyChanges, setOnlyChanges] = useState(true);
  if (busy) return <div className="empty">{busy}</div>;
  if (error) return <div className="empty error">{error}</div>;
  if (!c) return <div className="empty">Nothing compared yet. Use the Params tab for a what-if, or compare with a git revision.</div>;
  const steps = c.steps.filter((s) => {
    if (!onlyChanges) return true;
    if (s.status === "same" || s.status === "not run") return false;
    return record === null || s.status !== "changed" || s.structural.length > 0 || s.outputs.some((o) => o.changedRows.includes(record));
  });
  const count = (st: string) => c.steps.filter((s) => s.status === st).length;
  return (
    <div className="compare">
      <div className="summary">
        <strong><del>{c.a}</del> → <ins>{c.b}</ins></strong>
        <span className="muted"> · {c.rows} rows · {count("changed")} changed, {count("added")} added, {count("removed")} removed</span>
        <label className="right"><input type="checkbox" checked={onlyChanges} onChange={(e) => setOnlyChanges(e.target.checked)} /> only changes</label>
      </div>
      {(c.errors.a || c.errors.b) && (
        <div className="error">
          {c.errors.a && <div>{c.a}: {c.errors.a}</div>}
          {c.errors.b && <div>{c.b}: {c.errors.b}</div>}
        </div>
      )}
      <div className="muted">
        {c.firstDivergence ? (
          <>Values first diverge at <a onClick={() => onSelect(c.firstDivergence!)}>{c.firstDivergence}</a>.</>
        ) : (
          "Every step produces the same values."
        )}
      </div>
      <table>
        <tbody>
          {steps.map((s) => (
            <tr key={s.path} className={`step ${s.status}`}>
              <td><span className={`badge-status ${s.status}`}>{s.status}</span></td>
              <td>
                <a onClick={() => onSelect(s.path)}>{s.path}</a>
                {s.structural.map((x) => (
                  <span key={x} className="chip small">{x}</span>
                ))}
              </td>
              <td><Diffs diffs={s.outputs} record={record} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      <h4>Output</h4>
      {c.output.length ? <Diffs diffs={c.output} record={record} /> : <div className="muted">The outputs are identical.</div>}
    </div>
  );
}
