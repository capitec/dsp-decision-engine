import { useState } from "react";
import { fieldOf, formatValue, recordLabel, type CallNodeJson, type ColumnSummary, type RecordKey } from "./model/protocol";

interface Props {
  columns: ColumnSummary[] | null;
  record: number | null;
  keyCol: RecordKey;
  selected?: CallNodeJson;
  /** Column names in pipeline order: inputs as first read, then results as written. */
  order: string[];
  picked?: string;
  onPick: (name?: string) => void;
}

/** The column's type, and what its values measure: declared on a step, or guessed from the name ("?"). */
function TypeCell({ c }: { c: ColumnSummary }) {
  const f = fieldOf(c.name, c.value ?? c.preview[0]);
  const title = !f
    ? undefined
    : f.assumed
      ? `Shown as ${f.kind}, guessed from the name. Declare it on the step to be sure: Annotated[float, Money()] (or Percent(), Duration("months")) from decider.`
      : `Shown as ${f.kind}: declared on a step.`;
  return <td className="muted" title={title}>{c.dtype}{f ? ` · ${f.kind}${f.assumed ? "?" : ""}` : ""}</td>;
}

/** The searchable state: every column, its value (for the focused record), and how the selected step uses it. */
export function StateTable({ columns, record, keyCol, selected, order, picked, onPick }: Props) {
  const [q, setQ] = useState("");
  const [by, setBy] = useState<"changedBy" | "writtenBy">("changedBy");
  if (!columns) return <div className="empty">No session running. Use “Run flow” to see the state here.</div>;
  const reads = new Set(selected?.inputs ?? []);
  const writes = new Set(selected?.outputs ?? []);
  // The selected step's columns first, then pipeline order; anything else (pass-through columns) last.
  const rank = (name: string) => (reads.has(name) || writes.has(name) ? -1 : order.indexOf(name) === -1 ? order.length : order.indexOf(name));
  const rows = columns
    .filter((c) => c.name.includes(q) || c.producer.includes(q) || c.dtype.includes(q))
    .sort((a, b) => rank(a.name) - rank(b.name));
  const role = (name: string) => [reads.has(name) && "reads", writes.has(name) && "writes"].filter(Boolean).join(", ");
  const step = selected?.path.split("/").pop();
  return (
    <div className="state">
      <input placeholder="Filter by column, producing step or type" value={q} onChange={(e) => setQ(e.target.value)} autoFocus />
      <table>
        <thead>
          <tr>
            <th>column</th>
            <th>{record === null ? "first values" : recordLabel(record, keyCol)}</th>
            {step && <th title={`How ${selected!.path} uses the column`}>used by {step}</th>}
            <th>
              {record === null ? (
                "written by"
              ) : (
                <select aria-label="attribute values to" value={by} onChange={(e) => setBy(e.target.value as typeof by)} onClick={(e) => e.stopPropagation()}
                  title="Last changed by: the last step that gave this record a different value. Last written by: the last step that wrote it at all, even the same value again.">
                  <option value="changedBy">last changed by</option>
                  <option value="writtenBy">last written by</option>
                </select>
              )}
            </th>
            <th>type</th>
            {record === null && <th title="Null values across all records">nulls</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <tr key={c.name} className={`${role(c.name) ? "touched" : ""} ${picked === c.name ? "picked" : ""}`} onClick={() => onPick(picked === c.name ? undefined : c.name)}>
              <td>{c.name}</td>
              <td className="preview">
                {record === null ? `${c.preview.map((x) => formatValue(x)).join(", ")}${c.rows > c.preview.length ? ", …" : ""}` : formatValue(c.value)}
              </td>
              {step && <td className="role">{role(c.name)}</td>}
              <td className="muted" title={c[by] ?? c.producer}>{(c[by] ?? c.producer).split("/").pop()}{c.versions > 1 ? ` (${c.versions} versions)` : ""}</td>
              <TypeCell c={c} />
              {record === null && <td>{c.nulls || ""}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
