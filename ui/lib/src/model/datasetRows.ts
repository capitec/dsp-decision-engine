export type SortDir = "asc" | "desc";
export type Row = Record<string, unknown>;

/** The rows (with their index in `rows`) whose cells contain every column's filter text, sorted by one column. */
export function filterSort(
  rows: Row[],
  filters: Record<string, string>,
  sort: { column: string; dir: SortDir } | null,
): { row: Row; i: number }[] {
  let out = rows.map((row, i) => ({ row, i }));
  for (const [col, query] of Object.entries(filters)) {
    const q = query.trim().toLowerCase();
    if (q) out = out.filter(({ row }) => String(row[col] ?? "").toLowerCase().includes(q));
  }
  if (sort) {
    const sign = sort.dir === "asc" ? 1 : -1;
    out.sort((a, b) => {
      const av = a.row[sort.column], bv = b.row[sort.column];
      const cmp = typeof av === "number" && typeof bv === "number" ? av - bv : String(av ?? "").localeCompare(String(bv ?? ""));
      return sign * cmp;
    });
  }
  return out;
}
