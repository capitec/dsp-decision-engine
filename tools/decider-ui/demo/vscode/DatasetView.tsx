import { Fragment, useMemo, useState } from "react";
import type { MockDataset } from "./datasets";

// A stand-in for what clicking a dataset in the sidebar should open: a sortable, per-column
// filterable preview of its rows, a play icon to debug one, and a way to see a whole record as
// JSON — the shape a nested (struct/list) column needs, not just the flat ones.

type SortDir = "asc" | "desc";

/** A struct/list value as the cell shows it: a short summary, with the full JSON on hover. */
function cellText(value: unknown): { text: string; title?: string } {
  if (value == null) return { text: "" };
  if (Array.isArray(value)) {
    return value.length === 0
      ? { text: "[]" }
      : { text: `[${value.length}] ${JSON.stringify(value)}`.slice(0, 40) + (JSON.stringify(value).length > 30 ? "…" : ""), title: JSON.stringify(value, null, 2) };
  }
  if (typeof value === "object") {
    const json = JSON.stringify(value);
    return { text: `{…} ${json}`.slice(0, 40) + (json.length > 30 ? "…" : ""), title: JSON.stringify(value, null, 2) };
  }
  return { text: String(value) };
}

export function DatasetView({ dataset, onDebugRecord }: {
  dataset: MockDataset;
  onDebugRecord: (row: number) => void;
}) {
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [openFilterCol, setOpenFilterCol] = useState<string | null>(null);
  const [sort, setSort] = useState<{ column: string; dir: SortDir } | null>(null);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const toggleSort = (column: string) =>
    setSort((s) => {
      if (s?.column !== column) return { column, dir: "asc" };
      if (s.dir === "asc") return { column, dir: "desc" };
      return null;
    });

  const toggleExpanded = (i: number) =>
    setExpanded((e) => {
      const next = new Set(e);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });

  const rows = useMemo(() => {
    let indexed = dataset.rows.map((row, i) => ({ row, i }));
    for (const [col, query] of Object.entries(filters)) {
      if (!query.trim()) continue;
      const q = query.toLowerCase();
      indexed = indexed.filter(({ row }) => String(row[col] ?? "").toLowerCase().includes(q));
    }
    if (sort) {
      const { column, dir } = sort;
      indexed = [...indexed].sort((a, b) => {
        const av = a.row[column], bv = b.row[column];
        const cmp = typeof av === "number" && typeof bv === "number" ? av - bv : String(av ?? "").localeCompare(String(bv ?? ""));
        return dir === "asc" ? cmp : -cmp;
      });
    }
    return indexed;
  }, [dataset, filters, sort]);

  return (
    <div className="dataset-view">
      <div className="dataset-view-head">
        <strong>{dataset.name || "pasted rows"}</strong>
        <span className="muted">
          {dataset.rowCount > dataset.rows.length
            ? `previewing ${dataset.rows.length} of ${dataset.rowCount.toLocaleString()} rows`
            : `${dataset.rowCount.toLocaleString()} row${dataset.rowCount === 1 ? "" : "s"}`}
        </span>
        {dataset.fingerprint && <span className="muted small" title="content fingerprint">· {dataset.fingerprint}</span>}
      </div>
      <div className="dataset-table-wrap">
        <table className="dataset-table">
          <thead>
            <tr>
              <th className="dataset-col-gutter" />
              {dataset.columns.map(([name]) => (
                <th key={name}>
                  <span className="dataset-th-row">
                    <span className="dataset-th-name">{name}</span>
                    <button
                      className={sort?.column === name ? "dataset-icon-btn active" : "dataset-icon-btn"}
                      title={sort?.column === name ? `sorted ${sort.dir}ending — click to change` : "sort"}
                      onClick={() => toggleSort(name)}
                    >
                      {sort?.column === name ? (sort.dir === "asc" ? "▲" : "▼") : "⇕"}
                    </button>
                    <button
                      className={filters[name] ? "dataset-icon-btn active" : "dataset-icon-btn"}
                      title={filters[name] ? `filtering: "${filters[name]}"` : "filter"}
                      onClick={() => setOpenFilterCol((c) => (c === name ? null : name))}
                    >
                      ⚲
                    </button>
                  </span>
                  {openFilterCol === name && (
                    <input
                      className="dataset-col-filter"
                      autoFocus
                      type="search"
                      placeholder={`Filter ${name}…`}
                      value={filters[name] ?? ""}
                      onChange={(e) => setFilters((f) => ({ ...f, [name]: e.target.value }))}
                      onBlur={() => setOpenFilterCol(null)}
                    />
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ row, i }) => (
              <Fragment key={i}>
                <tr>
                  <td className="dataset-col-gutter">
                    <button className="dataset-expand-btn" title="View as JSON" onClick={() => toggleExpanded(i)}>
                      {expanded.has(i) ? "▾" : "▸"}
                    </button>
                    <button className="dataset-play-btn" title="Debug this record" onClick={() => onDebugRecord(i)}>▶</button>
                  </td>
                  {dataset.columns.map(([name]) => {
                    const { text, title } = cellText(row[name]);
                    return <td key={name} title={title}>{text}</td>;
                  })}
                </tr>
                {expanded.has(i) && (
                  <tr className="dataset-json-row">
                    <td colSpan={dataset.columns.length + 1}>
                      <pre className="dataset-json">{JSON.stringify(row, null, 2)}</pre>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={dataset.columns.length + 1} className="dataset-empty">No rows match the current filters</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
