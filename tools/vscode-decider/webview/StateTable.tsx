import { useState } from "react";
import type { CallNodeJson, ColumnSummary } from "../src/protocol";

interface Props {
  columns: ColumnSummary[] | null;
  record: number | null;
  selected?: CallNodeJson;
  picked?: string;
  onPick: (name?: string) => void;
}

/** The searchable state: every column, marked by how the selected node touches it. */
export function StateTable({ columns, record, selected, picked, onPick }: Props) {
  const [q, setQ] = useState("");
  if (!columns) return <div className="empty">No session running. Use “Run flow” to see the state here.</div>;
  const rows = columns.filter((c) => c.name.includes(q) || c.producer.includes(q) || c.dtype.includes(q));
  const reads = new Set(selected?.inputs ?? []);
  const writes = new Set(selected?.outputs ?? []);
  const role = (name: string) => [reads.has(name) && "reads", writes.has(name) && "writes"].filter(Boolean).join(", ");
  return (
    <div className="state">
      <input placeholder="filter columns, producers, dtypes" value={q} onChange={(e) => setQ(e.target.value)} autoFocus />
      <table>
        <thead>
          <tr>
            <th>column</th>
            <th>{selected ? `${selected.path.split("/").pop()}` : "role"}</th>
            <th>dtype</th>
            <th>nulls</th>
            <th>producer</th>
            <th>v</th>
            <th>{record === null ? "preview" : `record ${record}`}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <tr key={c.name} className={`${role(c.name) ? "touched" : ""} ${picked === c.name ? "picked" : ""}`} onClick={() => onPick(picked === c.name ? undefined : c.name)}>
              <td>{c.name}</td>
              <td className="role">{role(c.name)}</td>
              <td>{c.dtype}</td>
              <td>{c.nulls || ""}</td>
              <td className="muted">{c.producer}</td>
              <td>{c.versions}</td>
              <td className="preview">
                {record === null
                  ? `${c.preview.map((v) => JSON.stringify(v)).join(", ")}${c.rows > c.preview.length ? ", …" : ""}`
                  : JSON.stringify(c.value)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
