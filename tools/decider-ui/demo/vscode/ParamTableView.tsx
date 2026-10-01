import { Fragment, useMemo, useState } from "react";

// The full-screen view a table-shaped parameter (a decision table or scorecard's rows) expands
// into — same sortable, per-column filterable, row-expands-to-JSON shape as a dataset's preview,
// since a parameter table is exactly that kind of data. Unlike a dataset preview, every cell here
// is the actual value a run would use, so cells are editable.

export type ParamRow = Record<string, unknown>;

export function ParamTableView({ path, name, rows, onChange }: {
  path: string;
  name: string;
  rows: ParamRow[];
  onChange: (rows: ParamRow[]) => void;
}) {
  const columns = rows.length ? Object.keys(rows[0]) : [];
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [openFilterCol, setOpenFilterCol] = useState<string | null>(null);
  const [sort, setSort] = useState<{ column: string; dir: "asc" | "desc" } | null>(null);
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

  const view = useMemo(() => {
    let indexed = rows.map((row, i) => ({ row, i }));
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
  }, [rows, filters, sort]);

  const setCell = (i: number, col: string, raw: string) => {
    const prev = rows[i][col];
    const value = typeof prev === "number" && raw.trim() !== "" && !Number.isNaN(Number(raw)) ? Number(raw) : raw;
    onChange(rows.map((r, idx) => (idx === i ? { ...r, [col]: value } : r)));
  };

  return (
    <div className="dataset-view">
      <div className="dataset-view-head">
        <strong>{name}</strong>
        <span className="muted">{path === "shared" ? "shared · all steps" : path} · {rows.length} row{rows.length === 1 ? "" : "s"}</span>
      </div>
      <div className="dataset-table-wrap">
        <table className="dataset-table">
          <thead>
            <tr>
              <th className="dataset-col-gutter" />
              {columns.map((name) => (
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
            {view.map(({ row, i }) => (
              <Fragment key={i}>
                <tr>
                  <td className="dataset-col-gutter">
                    <button className="dataset-expand-btn" title="View as JSON" onClick={() => toggleExpanded(i)}>
                      {expanded.has(i) ? "▾" : "▸"}
                    </button>
                  </td>
                  {columns.map((col) => (
                    <td key={col}>
                      <input className="params-input" value={String(row[col] ?? "")} onChange={(e) => setCell(i, col, e.target.value)} />
                    </td>
                  ))}
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
            {rows.length === 0 && (
              <tr>
                <td colSpan={Math.max(columns.length, 1) + 1} className="dataset-empty">This table has no rows.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
