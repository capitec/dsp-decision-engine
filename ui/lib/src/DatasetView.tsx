import { Fragment, useMemo, useState } from "react";
import { filterSort, type Row, type SortDir } from "./model/datasetRows";

/** A struct/list value as the cell shows it: a short summary, with the full JSON on hover. */
function cellText(value: unknown): { text: string; title?: string } {
  if (value == null) return { text: "" };
  if (typeof value !== "object") return { text: String(value) };
  const json = JSON.stringify(value);
  if (Array.isArray(value) && value.length === 0) return { text: "[]" };
  const text = `${Array.isArray(value) ? `[${value.length}]` : "{…}"} ${json}`;
  return { text: text.length > 40 ? text.slice(0, 40) + "…" : text, title: JSON.stringify(value, null, 2) };
}

/**
 * A dataset's rows as a table: sort and filter on every column, a row's JSON on ▸, and ▶ on hover to debug it.
 * `onDebugRecord` gets the row's index in `rows`. `rowCount` is the dataset's size when `rows` is only its head.
 */
export function DatasetView({ name, columns, rows, rowCount, onDebugRecord }: {
  name: string;
  columns: [string, string][];
  rows: Row[];
  rowCount?: number | null;
  onDebugRecord?: (row: number) => void;
}) {
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [openFilterCol, setOpenFilterCol] = useState<string | null>(null);
  const [sort, setSort] = useState<{ column: string; dir: SortDir } | null>(null);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const toggleSort = (column: string) =>
    setSort((s) => (s?.column !== column ? { column, dir: "asc" } : s.dir === "asc" ? { column, dir: "desc" } : null));

  const toggleExpanded = (i: number) =>
    setExpanded((e) => {
      const next = new Set(e);
      if (!next.delete(i)) next.add(i);
      return next;
    });

  const shown = useMemo(() => filterSort(rows, filters, sort), [rows, filters, sort]);
  const truncated = rowCount != null && rowCount > rows.length;

  return (
    <div className="dataset-view">
      <div className="dataset-view-head">
        <strong>{name}</strong>
        <span className="muted">
          {truncated ? `first ${rows.length.toLocaleString()} of ${rowCount.toLocaleString()} rows` : `${rows.length.toLocaleString()} row${rows.length === 1 ? "" : "s"}`}
        </span>
        {truncated && <span className="muted small">· sort and filter apply to the loaded rows</span>}
      </div>
      <div className="dataset-table-wrap">
        <table className="dataset-table">
          <thead>
            <tr>
              <th className="dataset-col-gutter" />
              {columns.map(([col, dtype]) => (
                <th key={col}>
                  <span className="dataset-th-row">
                    <span className="dataset-th-name">{col}</span>
                    <button
                      className={sort?.column === col ? "dataset-icon-btn active" : "dataset-icon-btn"}
                      title={sort?.column === col ? `sorted ${sort.dir}ending — click to change` : "sort"}
                      onClick={() => toggleSort(col)}
                    >
                      {sort?.column === col ? (sort.dir === "asc" ? "▲" : "▼") : "⇕"}
                    </button>
                    <button
                      className={filters[col] ? "dataset-icon-btn active" : "dataset-icon-btn"}
                      title={filters[col] ? `filtering: "${filters[col]}"` : "filter"}
                      onClick={() => setOpenFilterCol((c) => (c === col ? null : col))}
                    >
                      ⚲
                    </button>
                  </span>
                  <span className="dataset-dtype">{dtype}</span>
                  {openFilterCol === col && (
                    <input
                      className="dataset-col-filter"
                      autoFocus
                      type="search"
                      placeholder={`Filter ${col}…`}
                      value={filters[col] ?? ""}
                      onChange={(e) => setFilters((f) => ({ ...f, [col]: e.target.value }))}
                      onBlur={() => setOpenFilterCol(null)}
                    />
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map(({ row, i }) => (
              <Fragment key={i}>
                <tr>
                  <td className="dataset-col-gutter">
                    {onDebugRecord && (
                      <button className="dataset-play-btn" title="Debug this record" onClick={() => onDebugRecord(i)}>▶</button>
                    )}
                    <button className="dataset-expand-btn" title="View as JSON" onClick={() => toggleExpanded(i)}>
                      {expanded.has(i) ? "▾" : "▸"}
                    </button>
                  </td>
                  {columns.map(([col]) => {
                    const { text, title } = cellText(row[col]);
                    return <td key={col} title={title}>{text}</td>;
                  })}
                </tr>
                {expanded.has(i) && (
                  <tr className="dataset-json-row">
                    <td colSpan={columns.length + 1}>
                      <pre className="dataset-json">{JSON.stringify(row, null, 2)}</pre>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
            {shown.length === 0 && (
              <tr>
                <td colSpan={columns.length + 1} className="dataset-empty">No rows match the current filters</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
