import { useState, type ReactNode } from "react";
import { paramChangeLines, paramReaders, same, type Comparison, type ValueDiff } from "../src/compare";
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
                step wrote <strong className="mono">{formatValue(s.b, d.name)}</strong> <span className="muted">(was {formatValue(s.a, d.name)})</span>
                {results && d.name in results.b && !(same(results.a[d.name]?.[s.row], s.a) && same(results.b[d.name]?.[s.row], s.b)) && (
                  <span> · final <strong className="mono">{formatValue(results.b[d.name]?.[s.row], d.name)}</strong> <span className="muted">(was {formatValue(results.a[d.name]?.[s.row], d.name)})</span></span>
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

/** The result columns that changed, for the records they changed on; the unchanged ones on request. */
function Results({ c, record }: { c: Comparison; record: number | null }) {
  const [showSame, setShowSame] = useState(false);
  const moved = (n: string, r: number) => !same(c.results.a[n]?.[r], c.results.b[n]?.[r]);
  const cols = Object.keys(c.results.b);
  const everyRow = Array.from({ length: c.rows }, (_, i) => i);
  const hit = everyRow.filter((r) => cols.some((n) => moved(n, r)));
  const changedCols = cols.filter((n) => hit.some((r) => moved(n, r)));
  // The decision reads first, changed or not: "still declined" and "now approved" are the point.
  const lead = cols.filter((n) => /^decision$/.test(n) && !changedCols.includes(n));
  const shownCols = showSame ? [...lead, ...changedCols, ...cols.filter((n) => !changedCols.includes(n) && !lead.includes(n))] : [...lead, ...changedCols];
  // One row per changed record, the focused one first; unchanged records on request.
  const rest = showSame ? [...hit, ...everyRow.filter((r) => !hit.includes(r))] : hit;
  const rows = record !== null ? [record, ...rest.filter((r) => r !== record)] : rest;
  if (!hit.length) return <div>No result changes for any of the {c.rows} records.</div>;
  return (
    <>
      <div className="muted small">
        {hit.length} of {c.rows} records changed.{record !== null && !hit.includes(record) ? ` ${recordLabel(record, c.key)} (focused) is unchanged; it is pinned first.` : ""}{" "}
        <label>
          <input type="checkbox" checked={showSame} onChange={(e) => setShowSame(e.target.checked)} /> also show the {c.rows - hit.length} unchanged records and {cols.length - changedCols.length} unchanged fields
        </label>
      </div>
      <div className="results-scroll">
        <table className="results">
          <thead>
            <tr>
              <th>record</th>
              {shownCols.map((n) => (
                <th key={n} className="mono">{n}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r} className={r === record ? "hit" : ""}>
                <td className="nowrap">{recordLabel(r, c.key)}</td>
                {shownCols.map((n) => {
                  const a = c.results.a[n]?.[r];
                  const b = c.results.b[n]?.[r];
                  return same(a, b) ? (
                    <td key={n} className="mono unchanged-cell" title="same in both runs">{formatValue(b, n)}</td>
                  ) : (
                    <td key={n} className="mono changed-cell"><s className="before">{formatValue(a, n)}</s> <span className="after">{formatValue(b, n)}</span></td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

// Steps named in the summary line; the step-by-step list below has them all.
const LIST = 8;

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
  const readers = paramReaders(c.paramsDocs?.b, c.sharedUsers);
  const readsChanged = (path: string) => readers.some((r) => r.steps.includes(path));
  const edited = (s: (typeof c.steps)[number]) => s.status !== "changed" || s.structural.length > 0 || s.paramChanges.length > 0 || docHas(s.path) || readsChanged(s.path);
  const label = (s: (typeof c.steps)[number]) => (edited(s) ? s.status : "affected");
  const outcomes = Object.keys(c.results.b);
  const movedOut = outcomes
    .map((n) => ({ n, k: Array.from({ length: c.rows }, (_, r) => r).filter((r) => !same(c.results.a[n]?.[r], c.results.b[n]?.[r])).length }))
    .filter((x) => x.k > 0);
  const stayed = outcomes.filter((n) => !movedOut.some((m) => m.n === n));
  const paramLines = [
    ...paramChangeLines(c.paramsDocs?.b, c.values?.a),
    ...c.steps.flatMap((s) => s.paramChanges.map((p) => `${s.path.split("/").pop()} ${p} (in its code)`)),
  ];
  // A changed param whose readers all wrote the same values: say so, or "nothing changed" reads as a bug.
  const idle = readers.filter((p) => p.steps.length && !p.steps.some((s) => c.steps.find((x) => x.path === s)?.outputs.length));
  return (
    <div className="compare">
      {back && <a className="back" onClick={back.go}>← Back to {back.label}</a>}
      <h3 className="compare-title">{c.b}</h3>
      {c.note && <div className="muted">{c.note}</div>}
      <div className="comparing">
        Compared with <strong>{c.a}</strong>
        <span className="muted"> · {c.rows} records · {count("changed")} steps changed{count("added") ? `, ${count("added")} added` : ""}{count("removed") ? `, ${count("removed")} removed` : ""}</span>
      </div>
      <div className="verdict">
        {movedOut.length === 0
          ? `No final output changes: every one of the ${c.rows} records ends the same.`
          : `${movedOut.map((m) => `${m.n} changes for ${m.k} of ${c.rows} records`).join("; ")}.`}
        {stayed.length > 0 && movedOut.length > 0 && stayed.length <= 8 && <span className="muted"> Unchanged: {stayed.join(", ")}.</span>}
      </div>
      {idle.map((p) => (
        <div key={p.param} className="note">
          <span className="mono">{p.param}</span> is read by{" "}
          {p.steps.slice(0, 6).map((s, i) => (
            <span key={s}>
              {i > 0 && ", "}
              <a onClick={() => onSelect(s)}>{s.split("/").pop()}</a>
            </span>
          ))}
          {p.steps.length > 6 && ` and ${p.steps.length - 6} more`}, but none of {p.steps.length === 1 ? "its" : "their"} outputs changed for any of the {c.rows} records.
        </div>
      ))}
      {c.steps.some((s) => s.status !== "same" && s.status !== "not run") && (
        <table className="changed-table">
          <thead>
            <tr>
              <th>step</th>
              <th>in</th>
              <th>what changed</th>
              <th>records</th>
            </tr>
          </thead>
          <tbody>
            {c.steps
              .filter((s) => s.status !== "same" && s.status !== "not run")
              // Edited steps first: they are the cause, the rest only follow from them.
              .sort((x, y) => Number(edited(y)) - Number(edited(x)))
              .slice(0, LIST)
              .map((s) => (
                <tr key={s.path}>
                  <td><a onClick={() => onSelect(s.path)}>{s.path.split("/").pop()}</a></td>
                  <td className="muted small">{s.path.split("/").slice(-3, -1).join("/")}</td>
                  <td>
                    {[
                      ...s.paramChanges,
                      ...readers.filter((r) => r.steps.includes(s.path)).map((r) => `${r.param} param`),
                      ...s.structural.filter((x) => x !== "params").map((x) => (x === "code" ? "code changed" : `${x} changed`)),
                      ...(s.status === "removed" ? ["skipped / removed"] : s.status === "added" ? ["added"] : []),
                    ].join(", ") || <span className="muted">follows from the above</span>}
                  </td>
                  <td className="mono">{new Set(s.outputs.flatMap((o) => o.changedRows)).size || "—"}</td>
                </tr>
              ))}
          </tbody>
        </table>
      )}
      {count("changed") + count("added") + count("removed") > LIST && (
        <div className="muted small">and {count("changed") + count("added") + count("removed") - LIST} more steps that follow from these; see Step by step below</div>
      )}
      {header}
      {(c.errors.a || c.errors.b) && (
        <div className="error">
          {c.errors.a && <div>{c.a}: {c.errors.a}</div>}
          {c.errors.b && <div>{c.b}: {c.errors.b}</div>}
        </div>
      )}
      {paramLines.length > 0 && (
        <>
          <h4>Changed params</h4>
          {paramLines.map((l) => (
            <div key={l} className="mono">{l}</div>
          ))}
        </>
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
      {withRevision && <div className="actions">{revisionButton}</div>}
    </div>
  );
}
