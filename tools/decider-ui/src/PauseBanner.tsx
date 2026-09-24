import { useState, type ReactNode } from "react";
import { editLabel, formatValue, recordLabel, type ColumnSummary, type FromUI, type RecordKey, type RunStatus, type Watch } from "./model/protocol";
import { hitText } from "./Controls";

interface Props {
  run: RunStatus;
  columns: ColumnSummary[] | null;
  keyCol: RecordKey;
  /** What the flow decides: its emitted values and what its top-level branches set. */
  outcome: string[];
  watches: Watch[];
  recordPicker: ReactNode;
  /** What the last skip, swap or re-run did. */
  note?: string;
  explained?: string;
  onExplain: (name?: string) => void;
  send: (m: FromUI) => void;
}

/** Where the run is paused, and what can be done from there: focus a record, explain a value, compare edits. */
export function PauseBanner({ run, columns, keyCol, outcome, watches, recordPicker, note, explained, onExplain, send }: Props) {
  const [hitsOpen, setHitsOpen] = useState(false);
  const current = run.current!;
  const edits = Object.entries(run.edits ?? {});
  const compareEdits = (only?: string) =>
    send({ type: "compareEdits", label: only ? editLabel(edits.find(([p]) => p === only)!) : edits.map(editLabel).join(", "), edits: run.edits!, path: only });
  // What the flow decides, for the focused record, as far as the run has got.
  const decided = run.record !== null && columns ? outcome.map((n) => columns.find((c) => c.name === n)).filter((c): c is ColumnSummary => !!c && c.value !== null && c.value !== undefined) : [];
  // A declined record has no offer: its amounts and rates are working values, so they stay out of the outcome.
  const declined = decided.some((c) => c.name === "decision" && c.value === "decline");
  const visible = decided.filter((c) => typeof c.value !== "boolean" && c.value !== "");
  const said = declined ? visible.filter((c) => typeof c.value === "string") : visible;
  const hit = run.hit;
  const hitRows = hit?.rows ?? [];
  return (
    <div className="pause-banner" title={current.path}>
      ⏸ Paused {current.when} <strong>{current.path.split("/").pop() || "the start"}</strong>
      {current.iteration ? <> in iteration <strong>{current.iteration}</strong></> : null}
      {columns && <> · focus {recordPicker}</>}
      {run.record !== null && columns && (
        <>
          {" "}
          <select
            aria-label="explain"
            title="How was a value computed for this record? Pick one to see its breakdown on the step that wrote it"
            value={explained ?? ""}
            onChange={(e) => onExplain(e.target.value || undefined)}
          >
            <option value="">explain a value…</option>
            {columns
              .filter((c) => c.producer !== "input")
              .map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} = {formatValue(c.value, c.name)}
                  {declined && typeof c.value === "number" && outcome.includes(c.name) ? " (no offer)" : ""}
                </option>
              ))}
          </select>
        </>
      )}
      {edits.length > 0 && (
        <>
          {" · "}
          <span className="edit-chip" title={edits.map(editLabel).join("\n")}>✎ {edits.length === 1 ? editLabel(edits[0]) : `${edits.length} edits`}</span>{" "}
          <button className="banner-button" title="Run the flow as started and as edited, start to end, and compare every result" onClick={() => compareEdits()}>
            {edits.length > 1 ? `Compare all ${edits.length} edits with start` : "Compare with start"}
          </button>
          {edits.length > 1 && (
            <select aria-label="compare one edit" value="" title="Compare the flow as started with only one of the edits" onChange={(e) => e.target.value && compareEdits(e.target.value)}>
              <option value="">…or just one edit</option>
              {edits.map((e) => (
                <option key={e[0]} value={e[0]}>{editLabel(e)}</option>
              ))}
            </select>
          )}
        </>
      )}
      {note && <div className="banner-note">{note}</div>}
      {hit && (
        <div className="banner-note hit">
          ⏸ Breakpoint: {hitText(hit, watches[hit.watch])}
          {hitRows.length > 0 && ": "}
          {hitRows.slice(0, hitsOpen ? undefined : 3).map((r, i) => (
            <span key={r}>
              {i > 0 && ", "}
              <a title="Focus this record" onClick={() => send({ type: "record", row: r })}>{recordLabel(r, keyCol)}</a> = {formatValue(hit.values?.[i], watches[hit.watch]?.name)}
            </span>
          ))}
          {hitRows.length > 3 && !hitsOpen && (
            <>
              {" "}
              <a onClick={() => setHitsOpen(true)}>and {hitRows.length - 3} more</a>
            </>
          )}
        </div>
      )}
      {said.length > 0 && (
        <div className="banner-note outcome">
          {`${run.finished ? "Outcome" : "Outcome so far"} for ${recordLabel(run.record!, keyCol)}: ${said.map((c) => `${c.name} = ${formatValue(c.value, c.name)}`).join(" · ")}${declined ? " · no offer is made" : ""}`}
        </div>
      )}
    </div>
  );
}
