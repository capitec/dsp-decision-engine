import { useState, type ReactNode } from "react";
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
  /** Set when the comparison was opened from somewhere to go back to, e.g. "6 scenarios". */
  back?: { label: string; go: () => void };
  /** Shown above the comparison, e.g. a scenario pager. */
  header?: ReactNode;
  /** Offer "Compare with a git revision…" (the Compare tab does; an inline scenario comparison doesn't). */
  withRevision?: boolean;
}

function Diffs({ diffs, record, keyCol, results }: { diffs: ValueDiff[]; record: number | null; keyCol: RecordKey; results?: Comparison["results"] }) {
  return (
    <table className="vdiffs">
      <tbody>
        {diffs.flatMap((d) =>
          d.samples.map((s, i) => (
            <tr key={`${d.name}-${s.row}`} className={record === s.row ? "hit" : ""}>
              <td className="mono">{i === 0 ? d.name : ""}</td>
              <td className="nowrap">{recordLabel(s.row, keyCol)}:</td>
              <td>
                step wrote <strong className="mono">{formatValue(s.b)}</strong> <span className="muted">(was {formatValue(s.a)})</span>
                {results && d.name in results.b && !(same(results.a[d.name]?.[s.row], s.a) && same(results.b[d.name]?.[s.row], s.b)) && (
                  <span> · final <strong className="mono">{formatValue(results.b[d.name]?.[s.row])}</strong> <span className="muted">(was {formatValue(results.a[d.name]?.[s.row])})</span></span>
                )}
                {i === 0 && d.changedRows.length > d.samples.length && <span className="muted"> · +{d.changedRows.length - d.samples.length} more records</span>}
              </td>
            </tr>
          )),
        )}
      </tbody>
    </table>
  );
}

/** Each result column, changed or not, with its values: the answer to "did the decision move?". */
function Results({ c, record }: { c: Comparison; record: number | null }) {
  const all = record === null ? Array.from({ length: c.rows }, (_, i) => i) : [record];
  const rows = all.slice(0, 4);
  // Changed outcomes first, then the rest with their values: "did the decision move?" needs both.
  const changed = (n: string) => rows.some((r) => !same(c.results.a[n]?.[r], c.results.b[n]?.[r]));
  const names = [...Object.keys(c.results.b).filter(changed), ...Object.keys(c.results.b).filter((n) => !changed(n))];
  return (
    <>
    {!names.some(changed) && <div>No result changes{record === null ? "" : ` for ${recordLabel(record, c.key)}`}.</div>}
    {names.length > 0 && (
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
                <td key={r} className="mono unchanged-cell" title="same in both runs">= {formatValue(b)}</td>
              ) : (
                <td key={r} className="mono changed-cell"><s className="before">{formatValue(a)}</s> <span className="after">{formatValue(b)}</span></td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
    )}
    {all.length > rows.length && <div className="muted">Showing {rows.length} of {all.length} records; focus a record to see it here.</div>}
    </>
  );
}

/** Two runs side by side, step by step, in execution order. */
export function Compare({ comparison: c, busy, error, record, onSelect, onCompareRevision, onOpenDiff, back, header, withRevision = true }: Props) {
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
  // A step is "edited" when its code or params differ; otherwise it only moved because its inputs did.
  const docHas = (path: string) => path.split("/").reduce<unknown>((d, part) => (d as Record<string, unknown> | undefined)?.[part], c.paramsDocs?.b) !== undefined;
  const edited = (s: (typeof c.steps)[number]) => s.status !== "changed" || s.structural.length > 0 || s.paramChanges.length > 0 || docHas(s.path);
  const label = (s: (typeof c.steps)[number]) => (edited(s) ? s.status : "affected");
  const outcomes = Object.keys(c.results.b);
  const movedOut = outcomes
    .map((n) => ({ n, k: Array.from({ length: c.rows }, (_, r) => r).filter((r) => !same(c.results.a[n]?.[r], c.results.b[n]?.[r])).length }))
    .filter((x) => x.k > 0);
  const stayed = outcomes.filter((n) => !movedOut.some((m) => m.n === n));
  return (
    <div className="compare">
      {back && <a className="back" onClick={back.go}>← Back to {back.label}</a>}
      {withRevision && <div className="actions top-actions">{revisionButton}</div>}
      <div className="comparing">
        Comparing <strong>{c.a}</strong> → <strong>{c.b}</strong>
        <span className="muted"> · {c.rows} records · {count("changed")} steps changed{count("added") ? `, ${count("added")} added` : ""}{count("removed") ? `, ${count("removed")} removed` : ""}</span>
      </div>
      <div className="verdict">
        {movedOut.length === 0
          ? "No final output changes."
          : `${movedOut.map((m) => `${m.n} changes for ${m.k} of ${c.rows} records`).join("; ")}.`}
        {stayed.length > 0 && movedOut.length > 0 && <span className="muted"> Unchanged: {stayed.join(", ")}.</span>}
      </div>
      {c.steps.some((s) => s.status !== "same" && s.status !== "not run") && (
        <div className="changed-list">
          Changed:{" "}
          {c.steps
            .filter((s) => s.status !== "same" && s.status !== "not run")
            .map((s, i) => (
              <span key={s.path}>
                {i > 0 && ", "}
                <a onClick={() => onSelect(s.path)}>{s.path.split("/").pop()}</a>
                <span className="muted"> ({[...s.paramChanges.map((p) => p.split(":")[0] + " param"), ...s.structural.filter((x) => x !== "params"), ...(s.outputs.length ? ["values"] : [])].join(", ") || s.status})</span>
              </span>
            ))}
        </div>
      )}
      {header}
      {(c.errors.a || c.errors.b) && (
        <div className="error">
          {c.errors.a && <div>{c.a}: {c.errors.a}</div>}
          {c.errors.b && <div>{c.b}: {c.errors.b}</div>}
        </div>
      )}
      {c.changedInputs.length > 0 && (
        <>
          <h4>Changed params and inputs</h4>
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
      {steps.length === 0 ? (
        <div className="muted">Every step produces the same values.</div>
      ) : (
        <div className="muted">What each changed step wrote, and the final value where a later step changed it again.</div>
      )}
      {steps.map((s) => (
        <div key={s.path} className={`step-diff ${s.status}`}>
          <div>
            <span className={`badge-status ${label(s)}`} title={label(s) === "affected" ? "Not edited: it changed because a value it reads changed" : undefined}>{label(s)}</span> <a onClick={() => onSelect(s.path)}>{s.path}</a>
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
          {s.outputs.length > 0 && <Diffs diffs={s.outputs} record={record} keyCol={c.key} results={c.results} />}
        </div>
      ))}
    </div>
  );
}
