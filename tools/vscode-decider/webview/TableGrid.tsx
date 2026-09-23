import { formatValue } from "../src/protocol";

interface Props {
  name: string;
  /** Column name -> polars dtype, e.g. `{"min_term": "Float64"}`. */
  columns: Record<string, string>;
  rows: Record<string, unknown>[];
  /** The rows the flow runs with, to mark edited cells. */
  base: Record<string, unknown>[];
  onChange: (rows: Record<string, unknown>[]) => void;
}

const numeric = (dtype: string) => /^(Float|Int|UInt)/.test(dtype);

/** Cells as typed (text while editing) converted to the column's type. */
export function tableRows(rows: Record<string, unknown>[], columns: Record<string, string>): Record<string, unknown>[] {
  return rows.map((r) => Object.fromEntries(Object.entries(r).map(([c, v]) => [c, typeof v === "string" ? parseCell(v, columns[c] ?? "") : v])));
}

function parseCell(text: string, dtype: string): unknown {
  if (text.trim() === "") return null;
  if (numeric(dtype)) {
    const n = Number(text.replace(/[\s,%]/g, "")) / (text.trim().endsWith("%") ? 100 : 1);
    return Number.isNaN(n) ? text : n;
  }
  return text;
}

const cellText = (v: unknown) => (v === null || v === undefined ? "" : String(v));

/** A lookup table's rows as an editable grid: one input per cell, edited cells marked. */
export function TableGrid({ name, columns, rows, base, onChange }: Props) {
  const cols = Object.keys(columns);
  // Cells hold the typed text until the document is built, so "0." survives typing "0.25".
  const set = (i: number, col: string, text: string) => onChange(rows.map((r, j) => (j === i ? { ...r, [col]: text } : r)));
  return (
    <table className="table-grid" aria-label={`${name} rows`}>
      <thead>
        <tr>
          <th className="muted small">#</th>
          {cols.map((c) => (
            <th key={c} title={columns[c]}>{c}</th>
          ))}
          <th />
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            <td className="muted small">{i + 1}</td>
            {cols.map((c) => {
              const was = base[i]?.[c];
              const edited = base[i] === undefined || cellText(was) !== cellText(tableRows([r], columns)[0][c]);
              return (
                <td key={c} className={edited ? "edited" : ""} title={edited && base[i] ? `was ${formatValue(was ?? null)}` : undefined}>
                  <input aria-label={`${name} row ${i + 1} ${c}`} className={numeric(columns[c]) ? "num" : ""} value={cellText(r[c])} onChange={(e) => set(i, c, e.target.value)} />
                </td>
              );
            })}
            <td>
              <button className="link small" title="Remove this row" onClick={() => onChange(rows.filter((_, j) => j !== i))}>✕</button>
            </td>
          </tr>
        ))}
        <tr>
          <td colSpan={cols.length + 2}>
            <button className="link small" onClick={() => onChange([...rows, Object.fromEntries(cols.map((c) => [c, null]))])}>+ add a row</button>
            {rows.length === 0 && <span className="muted small"> This table has no rows: every record gets the table's default.</span>}
          </td>
        </tr>
      </tbody>
    </table>
  );
}
