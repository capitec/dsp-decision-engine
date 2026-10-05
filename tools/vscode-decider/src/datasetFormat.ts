import * as path from "node:path";

export interface Dataset {
  name: string;
  type: string;
  key: string[] | null;
  path: string | null;
  error?: string;
}

export type Columns = [string, string][];

export const escapeHtml = (v: unknown) =>
  String(v ?? "").replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

/** What the tree shows for a dataset: `folder · key client_id`, and the context value menus match on. */
export function datasetLabel(d: Dataset): { description: string; contextValue: string } {
  return {
    description: [d.type, d.key?.length ? `key ${d.key.join(", ")}` : ""].filter(Boolean).join(" · "),
    contextValue: d.error ? "dataset-broken" : d.path ? "dataset dataset-local" : "dataset",
  };
}

/** A table of rows under a header of column names and dtypes. */
export function htmlTable(columns: Columns, rows: Record<string, unknown>[]): string {
  const head = columns.map(([n, t]) => `<th>${escapeHtml(n)}<br><small>${escapeHtml(t)}</small></th>`).join("");
  const body = rows.map((r) => `<tr>${columns.map(([n]) => `<td>${escapeHtml(r[n])}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

/** Project roots: the directory above each `configs/` among the given file paths. */
export const projectRoots = (files: string[]): string[] => [
  ...new Set(files.flatMap((f) => { const m = /^(.*?)[\\/]configs[\\/]/.exec(f); return m ? [m[1]] : []; })),
].sort();

export function isUnder(root: string, file: string): boolean {
  const r = path.relative(root, file);
  return !r.startsWith("..") && !path.isAbsolute(r);
}
