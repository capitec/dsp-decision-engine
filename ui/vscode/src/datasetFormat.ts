import * as path from "node:path";

export interface Dataset {
  name: string;
  type: string;
  key: string[] | null;
  path: string | null;
  error?: string;
}

export type Columns = [string, string][];

/** What the tree shows for a dataset: `folder · key client_id`, and the context value menus match on. */
export function datasetLabel(d: Dataset): { description: string; contextValue: string } {
  return {
    description: [d.type, d.key?.length ? `key ${d.key.join(", ")}` : ""].filter(Boolean).join(" · "),
    contextValue: d.error ? "dataset-broken" : d.path ? "dataset dataset-local" : "dataset",
  };
}

/**
 * The rows (with their index in `rows`) whose cells contain every column's filter text, sorted by one column.
 * Self-contained: the preview page runs its source.
 */
export function filterSort(
  rows: Record<string, unknown>[],
  filters: Record<string, string>,
  sort: { column: string; dir: "asc" | "desc" } | null,
): { row: Record<string, unknown>; i: number }[] {
  let out = rows.map((row, i) => ({ row, i }));
  for (const [col, query] of Object.entries(filters)) {
    const q = query.trim().toLowerCase();
    if (q) out = out.filter(({ row }) => String(row[col] ?? "").toLowerCase().includes(q));
  }
  if (sort) {
    const sign = sort.dir === "asc" ? 1 : -1;
    out.sort((a, b) => {
      const x = a.row[sort.column], y = b.row[sort.column];
      return sign * (typeof x === "number" && typeof y === "number" ? x - y : String(x ?? "").localeCompare(String(y ?? "")));
    });
  }
  return out;
}

/** The preview page: draws what the host posts as `{type: "dataset", name, columns, rows, rowCount}`; ▶ posts `{type: "debugRecord", row}`. */
export const datasetPage = (nonce: string) => `<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>
body{color:var(--vscode-editor-foreground);background:var(--vscode-editor-background);font-family:var(--vscode-editor-font-family);font-size:var(--vscode-editor-font-size)}
table{border-collapse:collapse}th,td{border:1px solid var(--vscode-panel-border);padding:2px 8px;text-align:left}
th{background:var(--vscode-editorWidget-background);cursor:pointer}small{color:var(--vscode-descriptionForeground);font-weight:normal}
th .filter{visibility:hidden;border:none;background:none;padding:0 2px;margin-left:4px;cursor:pointer;color:var(--vscode-icon-foreground);vertical-align:middle}
th:hover .filter,th .filter:focus,th.filtered .filter{visibility:visible}th.filtered .filter{color:var(--vscode-textLink-foreground)}
#pop{position:absolute;z-index:10;display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:8px;background:var(--vscode-editorWidget-background);border:1px solid var(--vscode-editorWidget-border,var(--vscode-panel-border));border-radius:3px;box-shadow:0 2px 8px var(--vscode-widget-shadow)}
#pop[hidden]{display:none}#pop .title{width:100%;font-weight:600}
#pop input{color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--vscode-input-border,var(--vscode-panel-border));padding:2px 4px}
#pop button{color:var(--vscode-button-secondaryForeground);background:var(--vscode-button-secondaryBackground);border:none;padding:3px 8px;cursor:pointer}
td.go{border:none;padding:0 2px;background:none}td.go button{visibility:hidden;border:none;background:none;color:var(--vscode-textLink-foreground);cursor:pointer}
tr:hover td.go button,td.go button:focus{visibility:visible}
</style></head><body><p id="info"></p><table><thead id="head"></thead><tbody id="body"></tbody></table><div id="pop" hidden></div>
<script nonce="${nonce}">
const vscode = acquireVsCodeApi();
const filterSort = ${filterSort.toString()};
let data, sort = null;
const filters = {};
const FUNNEL = '<svg width="12" height="12" viewBox="0 0 16 16"><path fill="currentColor" d="M1 2h14l-5.5 6.5V14l-3-1.5V8.5z"/></svg>';
const pop = document.getElementById("pop");
const close = () => (pop.hidden = true);
document.addEventListener("mousedown", (e) => { if (!pop.contains(e.target)) close(); });
const el = (tag, text, parent) => { const e = document.createElement(tag); if (text != null) e.textContent = text; parent?.append(e); return e; };
function rows() {
  const body = document.getElementById("body");
  body.replaceChildren();
  for (const { row, i } of filterSort(data.rows, filters, sort)) {
    const tr = el("tr", null, body);
    const go = el("td", null, tr);
    go.className = "go";
    const b = el("button", "\u25B6", go);
    b.title = "Debug this record";
    b.onclick = () => vscode.postMessage({ type: "debugRecord", row: i });
    for (const [n] of data.columns) el("td", row[n] == null ? "" : typeof row[n] === "object" ? JSON.stringify(row[n]) : String(row[n]), tr);
  }
}
function render() {
  const { name, columns, rowCount } = data;
  const n = data.rows.length, more = rowCount != null && rowCount > n;
  document.getElementById("info").textContent = more
    ? "first " + n + " of " + rowCount + " rows (sort and filter apply to the loaded rows)"
    : n + " rows";
  const head = document.getElementById("head");
  head.replaceChildren();
  const tr = el("tr", null, head);
  el("th", null, tr).style.cursor = "default";
  for (const [c, t] of columns) {
    const th = el("th", c + (sort?.column === c ? (sort.dir === "asc" ? " \u25B2" : " \u25BC") : ""), tr);
    if (filters[c]) th.className = "filtered";
    th.onclick = () => { sort = sort?.column !== c ? { column: c, dir: "asc" } : sort.dir === "asc" ? { column: c, dir: "desc" } : null; render(); };
    const f = el("button", null, th);
    f.className = "filter";
    f.title = filters[c] ? "Filtered: " + filters[c] : "Filter " + c;
    f.innerHTML = FUNNEL;
    f.onclick = (e) => { e.stopPropagation(); openFilter(c, th); };
    el("br", null, th);
    el("small", t, th);
  }
  rows();
}
function openFilter(c, th) {
  pop.replaceChildren();
  el("div", "Filter " + c, pop).className = "title";
  const input = el("input", null, pop);
  input.value = filters[c] ?? "";
  input.placeholder = "contains\u2026";
  const apply = () => { if (input.value.trim()) filters[c] = input.value; else delete filters[c]; render(); };
  input.oninput = apply;
  input.onkeydown = (e) => { if (e.key === "Enter" || e.key === "Escape") close(); };
  el("button", "Clear", pop).onclick = () => { input.value = ""; apply(); close(); };
  const r = th.getBoundingClientRect();
  pop.style.left = r.left + window.scrollX + "px";
  pop.style.top = r.bottom + window.scrollY + 2 + "px";
  pop.hidden = false;
  input.focus();
}
window.addEventListener("message", (e) => { if (e.data.type === "dataset") { data = e.data; render(); } });
vscode.postMessage({ type: "ready" });
</script></body></html>`;

/** Project roots: the directory above each `configs/` among the given file paths. */
export const projectRoots = (files: string[]): string[] => [
  ...new Set(files.flatMap((f) => { const m = /^(.*?)[\\/]configs[\\/]/.exec(f); return m ? [m[1]] : []; })),
].sort();

export function isUnder(root: string, file: string): boolean {
  const r = path.relative(root, file);
  return !r.startsWith("..") && !path.isAbsolute(r);
}
