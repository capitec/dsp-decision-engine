import { useState, type ReactNode } from "react";
import { paramChangeLines, paramReaders, same, type Comparison, type ValueDiff } from "../src/compare";
import { formatValue, recordLabel, type RecordKey } from "../src/protocol";
import { headline, ResultCards } from "./ResultCards";

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
  /** Focus a record in the paused debugger; absent when no session is running. */
  onFocus?: (row: number) => void;
  /** Offer "Compare with a git revision…" (the Compare tab does; an inline scenario comparison doesn't). */
  withRevision?: boolean;
}

function Diffs({ diffs, record, keyCol, results }: { diffs: ValueDiff[]; record: number | null; keyCol: RecordKey; results?: Comparison["results"] }) {
  const [all, setAll] = useState(false);
  // The first few records of each output are sampled; "and N more" fills in the rest from the final results.
  const rowsOf = (d: ValueDiff) => (all && results ? d.changedRows.map((row) => d.samples.find((s) => s.row === row) ?? { row, a: results.a[d.name]?.[row], b: results.b[d.name]?.[row] }) : d.samples);
  return (
    <table className="vdiffs">
      <tbody>
        {diffs.flatMap((d) =>
          rowsOf(d).map((s, i) => (
            <tr key={`${d.name}-${s.row}`} className={record === s.row ? "hit" : ""}>
              <td className="mono">{i === 0 ? d.name : ""}</td>
              <td className="nowrap">{recordLabel(s.row, keyCol)}:</td>
              <td>
                step wrote <strong className="mono">{formatValue(s.b, d.name)}</strong> <span className="muted">(was {formatValue(s.a, d.name)})</span>
                {results && d.name in results.b && !(same(results.a[d.name]?.[s.row], s.a) && same(results.b[d.name]?.[s.row], s.b)) && (
                  <span> · final <strong className="mono">{formatValue(results.b[d.name]?.[s.row], d.name)}</strong> <span className="muted">(was {formatValue(results.a[d.name]?.[s.row], d.name)})</span></span>
                )}
                {!all && i === d.samples.length - 1 && d.changedRows.length > d.samples.length && (
                  <a className="small" onClick={() => setAll(true)}> and {d.changedRows.length - d.samples.length} more</a>
                )}
              </td>
            </tr>
          )),
        )}
      </tbody>
    </table>
  );
}

/** The result columns that changed, for the records they changed on; the unchanged ones on request. */
/** Two runs side by side, step by step, in execution order. */
export function Compare({ comparison: c, busy, error, record, onSelect, onCompareRevision, onOpenDiff, back, header, withRevision = true, onFocus }: Props) {
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
  const focusChanged = record !== null && c.steps.some((s) => s.outputs.some((o) => o.changedRows.includes(record)));
  const steps = c.steps.filter((s) => {
    if (!onlyChanges) return true;
    if (s.status === "same" || s.status === "not run") return false;
    return !focusChanged || s.status !== "changed" || s.structural.length > 0 || s.outputs.some((o) => o.changedRows.includes(record!));
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
  const paramLines = [
    ...paramChangeLines(c.paramsDocs?.b, c.values?.a),
    ...c.steps.flatMap((s) => s.paramChanges.map((p) => `${s.path.split("/").pop()} ${p} (in its code)`)),
  ];
  // The edited steps are the causes; the rest only follow from them.
  const touched = c.steps.filter((s) => s.status !== "same" && s.status !== "not run");
  // Readers of a changed param that wrote the same values: one line, not rows of zeros.
  const idleReaders = [...new Set(readers.flatMap((r) => r.steps))].filter((p) => !touched.some((s) => s.path === p));
  const causes = touched.filter(edited);
  // When no approved applicant's offer moved, say where the change went instead.
  const decision = c.results.b.decision;
  const approved = decision ? decision.map((d, r) => (d !== "decline" ? r : -1)).filter((r) => r >= 0) : [];
  const touchedApproved = new Set(touched.flatMap((s) => s.outputs.flatMap((o) => o.changedRows)).filter((r) => approved.includes(r)));
  const whyNoOffer =
    decision && readers.length && touched.length && !touchedApproved.size
      ? `For the ${approved.length} approved applicants, none of the ${[...new Set(readers.flatMap((r) => r.steps))].length} steps that read ${readers.map((r) => r.param).join(", ")} changed a value; only declined records moved.`
      : null;
  const rowEdits = (param: string) =>
    paramChangeLines(c.paramsDocs?.b, c.values?.a)
      .filter((l) => l.startsWith(`${param} row`))
      .map((l) => `${l.split(":")[0].slice(param.length + 1)} edited`);
  const downstream = touched.length - causes.length;
  // A changed param whose readers all wrote the same values: say so, or "nothing changed" reads as a bug.
  const idle = readers.filter((p) => p.steps.length && !p.steps.some((s) => c.steps.find((x) => x.path === s)?.outputs.length));
  return (
    <div className="compare">
      {back && <a className="back" onClick={back.go}>← Back to {back.label}</a>}
      <div className="compare-top">
        <h3 className="compare-title">{c.b}</h3>
        {withRevision && <a className="small" onClick={onCompareRevision}>compare with a git revision…</a>}
      </div>
      {c.note && <div className="muted">{c.note}</div>}
      <div className="comparing">
        Compared with <strong>{c.a}</strong>
        <span className="muted"> · {c.rows} records · {count("changed")} steps changed{count("added") ? `, ${count("added")} added` : ""}{count("removed") ? `, ${count("removed")} removed` : ""}</span>
      </div>
      <div className="verdict">
        {headline(c) ?? (movedOut.length === 0
          ? `No final output changes: every one of the ${c.rows} records ends the same.`
          : `${movedOut.map((m) => `${m.n} changes for ${m.k} of ${c.rows} records`).join("; ")}.`)}
      </div>
      {whyNoOffer && <div className="note">{whyNoOffer}</div>}
      {movedOut.length === 0 && causes.length > 0 && (
        <div className="note">
          Nothing changed: with {causes.map((s) => s.path.split("/").pop()).join(" and ")} {causes.some((s) => s.status === "removed") ? "left out" : "changed"}, every record ends the same, so
          {causes.length === 1 ? " it makes" : " they make"} no difference for these {c.rows} records.
        </div>
      )}
      {!causes.length && idle.map((p) => (
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
      {header}
      {(c.errors.a || c.errors.b) && (
        <div className="error">
          {c.errors.a && <div>{c.a}: {c.errors.a}</div>}
          {c.errors.b && <div>{c.b}: {c.errors.b}</div>}
        </div>
      )}
      <h4>Results</h4>
      <ResultCards c={c} record={record} onFocus={onFocus} />
      {(paramLines.length > 0 || c.changedInputs.length > 0 || causes.length > 0) && <h4>What changed</h4>}
      {paramLines.map((l) => (
        <div key={l} className="mono">{l}</div>
      ))}
      {c.changedInputs.map((i) => (
        <div key={i.name} className="mono">{i.name} → {formatValue(i.after, i.name)} <span className="muted">for {i.scope}</span></div>
      ))}
      {causes.length > 0 && (
        <table className="changed-table">
          <thead>
            <tr>
              <th>step</th>
              <th>file</th>
              <th>what changed</th>
              <th>records changed</th>
            </tr>
          </thead>
          <tbody>
            {causes.map((s) => (
              <tr key={s.path}>
                <td><a onClick={() => onSelect(s.path)}>{s.path.split("/").pop()}</a></td>
                <td className="muted small mono" title={s.path}>{rowEdits(readers.find((r) => r.steps.includes(s.path))?.param ?? "").length ? "PARAMS" : s.where ?? "—"}</td>
                <td>
                  {[
                    ...s.paramChanges,
                    ...readers.filter((r) => r.steps.includes(s.path)).flatMap((r) => (rowEdits(r.param).length ? rowEdits(r.param) : [`reads ${r.param}`])),
                    ...s.structural.filter((x) => x !== "params").map((x) => (x === "code" ? "code changed" : `${x} changed`)),
                    ...(s.status === "removed" ? ["skipped / removed"] : s.status === "added" ? ["added"] : []),
                  ].join(", ")}
                  {s.structural.includes("code") && c.files && <> · <a onClick={() => onOpenDiff(s.path)}>view diff</a></>}
                </td>
                <td className="mono">{new Set(s.outputs.flatMap((o) => o.changedRows)).size}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {idleReaders.length > 0 && (
        <div className="muted small">
          {idleReaders.length} more step{idleReaders.length === 1 ? "" : "s"} read {readers.map((r) => r.param).join(", ")} but changed no record:{" "}
          {idleReaders.map((p, i) => (
            <span key={p}>
              {i > 0 && ", "}
              <a onClick={() => onSelect(p)}>{p.split("/").pop()}</a>
            </span>
          ))}
        </div>
      )}
      {downstream > 0 && <div className="muted small">and {downstream} downstream step{downstream === 1 ? "" : "s"} changed as a result: see Step by step below.</div>}
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
