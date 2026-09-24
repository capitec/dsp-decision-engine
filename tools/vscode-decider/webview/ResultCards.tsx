import { useState } from "react";
import { same, type Comparison } from "../src/compare";
import { formatValue, recordLabel } from "../src/protocol";

// Records shown before "show more": enough to scan, few enough to keep the steps in reach.
const FIRST = 12;

/** The records whose results changed, one card each: approved first, declined ones grouped after. */
export function ResultCards({ c, record, onFocus, onSelect }: { c: Comparison; record: number | null; onFocus?: (row: number) => void; onSelect?: (path: string) => void }) {
  const writer = (n: string) => [...c.steps].reverse().find((s) => s.outputs.some((o) => o.name === n))?.path;
  const [more, setMore] = useState(false);
  const moved = (n: string, r: number) => !same(c.results.a[n]?.[r], c.results.b[n]?.[r]);
  const cols = Object.keys(c.results.b);
  const hit = Array.from({ length: c.rows }, (_, r) => r).filter((r) => cols.some((n) => moved(n, r)));
  const decision = c.results.b.decision ? "decision" : null;
  // A declined record has no offer, so its changed rates and amounts are not a changed offer.
  const declined = (r: number) => decision !== null && c.results.b[decision][r] === "decline" && c.results.a[decision][r] === "decline";
  // A decline for a new reason is the policy change itself, so it gets a card like a changed offer.
  const reasonMoved = (r: number) => !!c.results.b.reason_code && moved("reason_code", r);
  const offered = hit.filter((r) => !declined(r) || reasonMoved(r));
  const noOffer = hit.filter((r) => declined(r) && !reasonMoved(r));
  const ordered = record !== null && offered.includes(record) ? [record, ...offered.filter((r) => r !== record)] : offered;
  if (!hit.length) return <div>No result changes for any of the {c.rows} records.</div>;
  const card = (r: number) => (
    <div key={r} className={`result-card ${r === record ? "hit" : ""}`}>
      <div className="result-head">
        {onFocus && r !== record ? (
          <a title="Focus this record in the debugger" onClick={() => onFocus(r)}><strong>{recordLabel(r, c.key)}</strong></a>
        ) : (
          <strong>{recordLabel(r, c.key)}</strong>
        )}
        {decision && (
          <span className={`decision ${c.results.b[decision][r]}`}>
            {moved(decision, r) ? `${formatValue(c.results.a[decision][r])} → ${formatValue(c.results.b[decision][r])}` : formatValue(c.results.b[decision][r])}
          </span>
        )}
        {r === record && <span className="muted small"> (focused)</span>}
      </div>
      {cols
        .filter((n) => n !== decision && moved(n, r) && (!declined(r) || n === "reason_code"))
        .map((n) => (
          <div key={n} className="result-line">
            <span className="mono">{n}</span> <s className="before">{formatValue(c.results.a[n]?.[r], n)}</s> → <strong className="after">{formatValue(c.results.b[n]?.[r], n)}</strong>
            {onSelect && writer(n) && (
              <a className="small why" title={`Show ${writer(n)!.split("/").pop()}, the step that last wrote ${n}`} onClick={() => onSelect(writer(n)!)}> why?</a>
            )}
          </div>
        ))}
      {declined(r) && cols.some((n) => n !== decision && n !== "reason_code" && moved(n, r)) && (
        <>
          <div className="result-line muted">internal values (not offered):</div>
          {cols
            .filter((n) => n !== decision && n !== "reason_code" && moved(n, r))
            .map((n) => (
              <div key={n} className="result-line internal">
                <span className="mono">{n}</span> {formatValue(c.results.a[n]?.[r], n)} → {formatValue(c.results.b[n]?.[r], n)}
              </div>
            ))}
        </>
      )}
    </div>
  );
  return (
    <>
      {!offered.length && noOffer.length > 0 && <div>No offer changed.</div>}
      {record !== null && !hit.includes(record) && <div className="muted">{recordLabel(record, c.key)} (focused) is unchanged{hit.length ? "; the records below changed" : ""}.</div>}

      {(more ? ordered : ordered.slice(0, FIRST)).map(card)}
      {ordered.length > FIRST && (
        <button className="link" onClick={() => setMore(!more)}>{more ? "show fewer" : `show all ${ordered.length} changed records`}</button>
      )}
      {noOffer.length > 0 && (
        <details className="no-offer">
          <summary>
            Declined either way ({noOffer.length}): offer numbers moved but no offer is made
          </summary>
          {noOffer.map(card)}
        </details>
      )}
    </>
  );
}

/** "Decisions changed: 0 · offers changed for 3 approved applicants · 2 declined records changed values only". */
export function headline(c: Comparison): string | null {
  const b = c.results.b.decision;
  if (!b) return null;
  const a = c.results.a.decision ?? [];
  const cols = Object.keys(c.results.b).filter((n) => n !== "decision");
  const rows = Array.from({ length: c.rows }, (_, r) => r);
  const decided = rows.filter((r) => !same(a[r], b[r]));
  const changed = rows.filter((r) => cols.some((n) => !same(c.results.a[n]?.[r], c.results.b[n]?.[r])));
  const offers = changed.filter((r) => b[r] !== "decline" || a[r] !== "decline");
  const reasonMoved = (r: number) => !!c.results.b.reason_code && !same(c.results.a.reason_code?.[r], c.results.b.reason_code[r]);
  const valuesOnly = changed.filter((r) => b[r] === "decline" && a[r] === "decline" && !reasonMoved(r)).length;
  const flips = decided.map((r) => `${formatValue(a[r])} → ${formatValue(b[r])}`);
  const counts = [...new Set(flips)].map((f) => `${flips.filter((x) => x === f).length} ${f}`).join(", ");
  const reason = c.results.b.reason_code;
  const reasons = reason ? rows.filter((r) => a[r] === "decline" && b[r] === "decline" && !same(c.results.a.reason_code?.[r], reason[r])).length : 0;
  return [
    `Decisions changed: ${decided.length}${decided.length ? ` (${counts})` : ""}`,
    `Offers changed: ${offers.length}`,
    reasons ? `Declines with a new reason: ${reasons}` : "",

  ]
    .filter(Boolean)
    .join(" · ");
}
