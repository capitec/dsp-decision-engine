import { formatValue } from "../src/protocol";

interface Props {
  name: string;
  /** Column name -> polars dtype, e.g. `{"min_term": "Float64"}`. */
  columns: Record<string, string>;
  rows: Record<string, unknown>[];
  /** The rows the flow runs with, to mark edited cells. */
  base: Record<string, unknown>[];
  onChange: (rows: Record<string, unknown>[]) => void;
  /** How a record matches a row, to label band edges: `{type: "between", lower_bound_column, …}`. */
  expression?: Record<string, unknown> | null;
}

/** "min_term (from, incl.)" for a band's edges, else the column's name. */
function header(col: string, expr?: Record<string, unknown> | null): string {
  if (expr?.type !== "between") return col;
  const upperInclusive = expr.mode === "upper_inclusive";
  if (col === expr.lower_bound_column) return `${col} (from, ${upperInclusive ? "excl." : "incl."})`;
  if (col === expr.upper_bound_column) return `${col} (to, ${upperInclusive ? "incl." : "excl."})`;
  return col;
}

// Rates, loadings and discounts read as percentages; typing "24.5%" stores 0.245.
const isPercent = (col: string, rows: Record<string, unknown>[]) =>
  /rate|loading|discount/.test(col) && rows.every((r) => typeof r[col] !== "number" || Math.abs(r[col] as number) < 1);

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

const cellText = (v: unknown, percent = false) =>
  v === null || v === undefined ? "" : percent && typeof v === "number" ? `${Number((v * 100).toFixed(4))}%` : String(v);

/** A lookup table's rows as an editable grid: one input per cell, edited cells marked. */
export function TableGrid({ name, columns, rows, base, onChange, expression }: Props) {
  const keys = new Set([expression?.lower_bound_column, expression?.upper_bound_column, expression?.value_column].filter(Boolean));
  const cols = [...Object.keys(columns).filter((c) => !keys.has(c)), ...Object.keys(columns).filter((c) => keys.has(c))];
  const percent = Object.fromEntries(cols.map((c) => [c, isPercent(c, base)]));
  // Cells hold the typed text until the document is built, so "0." survives typing "0.25".
  const set = (i: number, col: string, text: string) => onChange(rows.map((r, j) => (j === i ? { ...r, [col]: text } : r)));
  return (
    <table className="table-grid" aria-label={`${name} rows`}>
      <thead>
        <tr>
          <th className="muted small">#</th>
          {cols.map((c) => (
            <th key={c} title={columns[c]}>{header(c, expression)}</th>
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
                <td key={c} className={edited ? "edited" : ""} title={edited && base[i] ? `was ${cellText(was ?? null, percent[c]) || formatValue(null)}` : undefined}>
                  <input aria-label={`${name} row ${i + 1} ${c}`} className={`${numeric(columns[c]) ? "num" : ""} ${keys.has(c) ? "key" : ""}`} value={cellText(r[c], percent[c])} onChange={(e) => set(i, c, e.target.value)} />
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
