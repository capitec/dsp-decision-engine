import { useState } from "react";
import type { CallNodeJson, ColumnSummary } from "../src/protocol";

interface Props {
  columns: ColumnSummary[] | null;
  selected?: CallNodeJson;
  picked?: string;
  onPick: (name?: string) => void;
}

/** The searchable state: every column, marked by how the selected node touches it. */
export function StateTable({ columns, selected, picked, onPick }: Props) {
  const [q, setQ] = useState("");
  if (!columns) return <div className="empty">No session running. Use “Run flow” to see the state here.</div>;
  const rows = columns.filter((c) => c.name.includes(q) || c.producer.includes(q) || c.dtype.includes(q));
  const role = (name: string) =>
    selected?.inputs.includes(name) && selected?.outputs.includes(name) ? "reads, writes" : selected?.inputs.includes(name) ? "reads" : selected?.outputs.includes(name) ? "writes" : "";
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
            <th>preview</th>
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
              <td className="preview">{c.preview.map((v) => JSON.stringify(v)).join(", ")}{c.rows > c.preview.length ? ", …" : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
