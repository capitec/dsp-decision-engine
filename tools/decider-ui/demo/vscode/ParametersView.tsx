import { useEffect, useState } from "react";
import type { DescribeResult, ParamInfo } from "../../src/model/protocol";

// A stand-in for a tab that lists every declared param across the flow — its default, its
// declared validation (bounds, required) — with an editable column for what a run would use
// instead. Nothing here recomputes the flow; it only collects the values. When a node is
// selected (in the graph or the sidebar tree, either feeds the same selection), it narrows to
// that node's own params plus the shared ones — a param belonging to some other step is noise
// once you're looking at one node. While a debug session is paused, editing reads as a live
// change to that run rather than something that waits for the next one.

export type ParamOverrides = Record<string, Record<string, string>>;

function bounds(p: ParamInfo): string | null {
  const parts: string[] = [];
  if (p.ge !== undefined) parts.push(`≥ ${p.ge}`);
  if (p.gt !== undefined) parts.push(`> ${p.gt}`);
  if (p.le !== undefined) parts.push(`≤ ${p.le}`);
  if (p.lt !== undefined) parts.push(`< ${p.lt}`);
  if (p.required) parts.push("required");
  return parts.length ? parts.join(", ") : null;
}

/** Why `value` wouldn't be accepted for `p`, or `null` when it's fine (an empty value just means "use the default"). */
function problemWith(value: string, p: ParamInfo): string | null {
  if (isTable(p)) return null;
  if (value.trim() === "") return p.required ? "required — no default to fall back to" : null;
  if (!p.type.includes("int") && !p.type.includes("float")) return null;
  const n = Number(value);
  if (Number.isNaN(n)) return `not a number`;
  if (p.ge !== undefined && n < p.ge) return `must be ≥ ${p.ge}`;
  if (p.gt !== undefined && n <= p.gt) return `must be > ${p.gt}`;
  if (p.le !== undefined && n > p.le) return `must be ≤ ${p.le}`;
  if (p.lt !== undefined && n >= p.lt) return `must be < ${p.lt}`;
  return null;
}

/** "shared" isn't a step — it's the one bucket in the params map with no node of its own. */
function stepLabel(path: string, pipeline: string): string {
  return path === "shared" ? "shared · all steps" : path || pipeline;
}

function isTable(p: ParamInfo): boolean {
  return p.type === "table" || Array.isArray(p.default);
}

export function ParametersView({ describe, overrides, onChange, onRun, focusPath, debugging, pausedAt, tableRowCounts, onOpenTable }: {
  describe: DescribeResult;
  overrides: ParamOverrides;
  onChange: (path: string, name: string, value: string) => void;
  onRun: () => void;
  /** The selected node's path, shared with the graph and the sidebar tree — narrows the list to it. */
  focusPath?: string;
  /** A debug session is paused: editing applies to it live, instead of waiting for the next run. */
  debugging?: boolean;
  /** The step the run is paused at — an edit applies from here on, not retroactively. */
  pausedAt?: string | null;
  /** A table param's row count once its full view has been opened and possibly edited there. */
  tableRowCounts: Record<string, Record<string, number>>;
  onOpenTable: (path: string, name: string) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const [query, setQuery] = useState("");
  // A fresh selection re-engages the filter, even if "show all" was picked for the last one.
  useEffect(() => setShowAll(false), [focusPath]);

  const allRows = Object.entries(describe.params ?? {}).flatMap(([path, byName]) =>
    Object.entries(byName).map(([name, info]) => ({ path, name, info })),
  );
  const scoped = focusPath !== undefined && !showAll;
  let rows = scoped ? allRows.filter((r) => r.path === focusPath || r.path === "shared") : allRows;
  if (query.trim()) {
    const q = query.toLowerCase();
    rows = rows.filter((r) => r.name.toLowerCase().includes(q) || r.path.toLowerCase().includes(q));
  }

  const changed = allRows.filter(({ path, name }) => (overrides[path]?.[name] ?? "") !== "").length;
  const problems = allRows.filter(({ path, name, info }) => problemWith(overrides[path]?.[name] ?? "", info) !== null).length;

  return (
    <div className="params-view">
      <div className="params-view-head">
        <strong>Parameters</strong>
        {scoped ? (
          <>
            <span className="muted">scoped to <span className="mono">{stepLabel(focusPath, describe.pipeline)}</span></span>
            <button className="link-btn" onClick={() => setShowAll(true)}>show all</button>
          </>
        ) : (
          focusPath !== undefined && <button className="link-btn" onClick={() => setShowAll(false)}>scope to <span className="mono">{stepLabel(focusPath, describe.pipeline)}</span></button>
        )}
        <span className="muted">{changed ? `${changed} overridden` : "defaults shown"}</span>
        {problems > 0 && <span className="params-problem-count">{problems} invalid</span>}
        <input
          className="dataset-col-filter params-search"
          type="search"
          placeholder="Search params…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>
      <div className="dataset-table-wrap">
        <table className="params-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>Param</th>
              <th>Type</th>
              <th>Default</th>
              <th>Validation</th>
              <th>{debugging ? "Live value" : "Value for next run"}</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={6} className="dataset-empty">
                  {allRows.length === 0
                    ? "This flow declares no parameters."
                    : query
                    ? `No parameters match "${query}".`
                    : `${stepLabel(focusPath ?? "", describe.pipeline)} has no parameters of its own.`}
                </td>
              </tr>
            )}
            {rows.map(({ path, name, info }) => {
              const table = isTable(info);
              const value = overrides[path]?.[name] ?? "";
              const problem = problemWith(value, info);
              const rowCount = tableRowCounts[path]?.[name] ?? (Array.isArray(info.default) ? info.default.length : 0);
              return (
                <tr key={`${path}.${name}`}>
                  <td className="mono">{stepLabel(path, describe.pipeline)}</td>
                  <td>{name}</td>
                  <td className="muted">{info.type}</td>
                  <td className="muted">{table ? `[${(info.default as unknown[]).length} rows]` : String(info.default ?? "—")}</td>
                  <td className="muted">{bounds(info) ?? "—"}</td>
                  <td>
                    {table ? (
                      <button className="link-btn" onClick={() => onOpenTable(path, name)} title="Open as a full table">
                        [{rowCount} rows] ⤢ expand
                      </button>
                    ) : (
                      <input
                        className={problem ? "params-input invalid" : "params-input"}
                        placeholder={String(info.default ?? "")}
                        value={value}
                        title={problem ?? undefined}
                        onChange={(e) => onChange(path, name, e.target.value)}
                      />
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {debugging ? (
        <p className="params-live-note">
          Editing a paused run — changes apply live, from {pausedAt ? <span className="mono">{pausedAt}</span> : "here"} onward;
          steps already run keep the value they ran with.
        </p>
      ) : (
        <button className="dataset-debug-btn params-run-btn" disabled={problems > 0} onClick={onRun}>▶ Run pipeline</button>
      )}
    </div>
  );
}
