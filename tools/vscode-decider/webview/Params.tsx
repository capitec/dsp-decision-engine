import { useState } from "react";
import { docWithDefaults, paramChangeLines, same } from "../src/compare";
import { formatValue, recordLabel, type ParamInfo, type RecordKey } from "../src/protocol";
import { TableGrid, tableRows } from "./TableGrid";

interface Props {
  schema: Record<string, Record<string, ParamInfo>>;
  /** The values the flow runs with (the module's PARAMS); a param missing here runs with its default. */
  values: Record<string, unknown>;
  inputColumns: string[];
  record: number | null;
  keyCol: RecordKey;
  sessionRunning: boolean;
  onWhatIf: (params: unknown, overrides: Record<string, unknown>, row: number | null, label: string) => void;
  onRestart: (params: unknown) => void;
  onSelectStep: (path: string) => void;
  /** Lookup table param -> how a record matches its rows. */
  tables?: Record<string, Record<string, unknown>>;
}

type Edits = Record<string, string>;
const OPEN_ALL = 20;
type Schema = Props["schema"];

/** The value a param runs with: from `values` (nested by path) if set there, else its default. */
export function currentValue(values: Record<string, unknown>, path: string, name: string, info: ParamInfo): unknown {
  let at: unknown = values;
  for (const part of [...path.split("/"), name]) at = at && typeof at === "object" ? (at as Record<string, unknown>)[part] : undefined;
  return at === undefined ? info.default ?? null : at;
}

const shown = (v: unknown, name?: string) => (Array.isArray(v) ? JSON.stringify(v) : formatValue(v, name));

/** An edit's value: a table's typed cells converted to their columns' types, anything else parsed by type. */
function editValue(text: string, info: ParamInfo): unknown {
  const v = parseValue(text, info.type);
  return info.type === "table" && Array.isArray(v) ? tableRows(v, info.schema as Record<string, string>) : v;
}

/** A params document holding only the values that differ from what the flow runs with, nested by path. */
export function paramsDocument(schema: Schema, edits: Edits, values: Record<string, unknown> = {}): Record<string, unknown> {
  const doc: Record<string, unknown> = {};
  for (const [key, text] of Object.entries(edits)) {
    const [path, name] = key.split("|");
    const info = schema[path]?.[name];
    if (text.trim() === "" || !info || same(editValue(text, info), currentValue(values, path, name, info))) continue;
    let target = doc as Record<string, Record<string, unknown>>;
    for (const part of path.split("/")) target = (target[part] ??= {}) as Record<string, Record<string, unknown>>;
    (target as Record<string, unknown>)[name] = editValue(text, info);
  }
  return doc;
}

export function parseValue(text: string, type?: string): unknown {
  if (/^\s*-?[\d.]+\s*%\s*$/.test(text)) return Number(text.replace("%", "")) / 100;
  if (type === "float" || type === "int" || type === "number" || type === "integer") {
    const n = Number(text);
    if (!Number.isNaN(n)) return n;
  }
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function bounds(info: ParamInfo): string {
  const parts = [info.ge !== undefined && `≥ ${info.ge}`, info.gt !== undefined && `> ${info.gt}`, info.le !== undefined && `≤ ${info.le}`, info.lt !== undefined && `< ${info.lt}`];
  return parts.filter(Boolean).join(", ");
}

/** Params grouped by the flow that holds their step; shared params first. */
export function groupParams(schema: Schema): [string, string[]][] {
  const groups = new Map<string, string[]>();
  for (const path of Object.keys(schema)) {
    const group = path === "shared" ? "shared" : path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "";
    groups.set(group, [...(groups.get(group) ?? []), path]);
  }
  return [...groups].sort(([a], [b]) => (a === "shared" ? -1 : b === "shared" ? 1 : 0));
}

const words = (q: string) => q.toLowerCase().replace(/_/g, " ").split(/\s+/).filter(Boolean);
const matches = (text: string, ws: string[]) => ws.every((w) => text.toLowerCase().replace(/_/g, " ").includes(w));

/** Every param with the value the flow runs with; change some, then compare, or restart the debug run with them. */
export function Params({ schema, values, inputColumns, record, keyCol, sessionRunning, onWhatIf, onRestart, onSelectStep, tables = {} }: Props) {
  const [edits, setEdits] = useState<Edits>({});
  const [overrides, setOverrides] = useState<{ column: string; value: string }[]>([]);
  const [scope, setScope] = useState<"record" | "all">("record");
  const [query, setQuery] = useState("");
  const [changedOnly, setChangedOnly] = useState(false);
  // A short list opens whole; a long one opens only the shared tables and rates.
  const [open, setOpen] = useState<Set<string>>(() => new Set(Object.keys(schema).length <= OPEN_ALL ? groupParams(schema).map(([g]) => g) : ["shared"]));
  const doc = paramsDocument(schema, edits, values);
  const override = Object.fromEntries(overrides.filter((o) => o.column && o.value !== "").map((o) => [o.column, parseValue(o.value, "number")]));
  const row = scope === "record" && record !== null ? record : null;
  const who = row === null ? "every record" : recordLabel(row, keyCol);
  const changes = [...paramChangeLines(doc, docWithDefaults(schema, values)), ...Object.entries(override).map(([k, v]) => `${k} = ${formatValue(v)} for ${who}`)];
  const ws = words(query);
  const total = Object.values(schema).reduce((n, ps) => n + Object.keys(ps).length, 0);

  const isEdited = (path: string, name: string) => {
    const text = edits[`${path}|${name}`];
    const info = schema[path][name];
    return text !== undefined && !same(editValue(text, info), currentValue(values, path, name, info));
  };
  const visible = (path: string, name: string) => (!changedOnly || isEdited(path, name)) && (!ws.length || matches(`${path} ${name}`, ws));

  const paramRow = (path: string, name: string, info: ParamInfo) => {
    const key = `${path}|${name}`;
    const base = currentValue(values, path, name, info);
    const edited = isEdited(path, name);
    const usedBy = (info.used_by as string[] | undefined) ?? [];
    const label = path === "shared" ? name : `${path.split("/").pop()} · ${name}`;
    const reset = <button className="link" onClick={() => setEdits({ ...edits, [key]: shown(base, name) })}>undo</button>;
    if (info.type === "table") {
      const rows = (edits[key] !== undefined ? parseValue(edits[key]) : base) as Record<string, unknown>[] | null;
      return (
        <tr key={key} className={edited ? "edited" : ""}>
          <td className="name" colSpan={3}>
            <div>
              <strong>{label}</strong>{" "}
              <span className="muted small">
                lookup table
                {usedBy.map((u) => (
                  <span key={u}> · <a title={u} onClick={() => onSelectStep(u)}>looked up in {u.split("/").slice(-3, -1).join("/")}</a></span>
                ))}
              </span>{" "}
              {edited && reset}
            </div>
            <TableGrid name={name} columns={info.schema as Record<string, string>} rows={rows ?? []} base={(base as Record<string, unknown>[] | null) ?? []} expression={tables[name]} onChange={(r) => setEdits({ ...edits, [key]: JSON.stringify(r) })} />
          </td>
        </tr>
      );
    }
    const text = edits[key] ?? shown(base, name);
    // Filtered down, a shared param's readers show as chips under it; otherwise behind a link.
    const chips = ws.length > 0 && usedBy.length > 0;
    return [
      <tr key={key} className={edited ? "edited" : ""}>
        <td className="name" title={path}>{label}</td>
        <td>
          <input aria-label={`${path} ${name}`} value={text} onChange={(e) => setEdits({ ...edits, [key]: e.target.value })} />
        </td>
        <td className="muted small">
          {edited ? <>was {shown(base, name)} · {reset}</> : bounds(info)}
          {usedBy.length > 0 && !chips && (
            <details className="used-by">
              <summary>used by {usedBy.length} step{usedBy.length === 1 ? "" : "s"}</summary>
              {usedBy.map((u) => (
                <div key={u}>
                  <a title={u} onClick={() => onSelectStep(u)}>{u.split("/").pop()}</a>
                </div>
              ))}
            </details>
          )}
        </td>
      </tr>,
      chips && (
        <tr key={`${key}-used`} className="used-by-row">
          <td colSpan={3}>
            <div className="muted">read by {usedBy.length} step{usedBy.length === 1 ? "" : "s"}:</div>
            {usedBy.map((u) => (
              <button key={u} className="chip" title={`${u}: show it in the graph`} onClick={() => onSelectStep(u)}>
                {u.split("/").pop()} <span className="muted small">{u.split("/").slice(-3, -2)}</span>
              </button>
            ))}
          </td>
        </tr>
      ),
    ];
  };

  return (
    <div className="params">
      <section className="params-bar">
        <input aria-label="Filter params" placeholder={`Filter ${total} params by step or name, e.g. repo_rate or base_rates`} value={query} onChange={(e) => setQuery(e.target.value)} />
        <label><input type="checkbox" checked={changedOnly} onChange={(e) => setChangedOnly(e.target.checked)} /> changed only</label>
      </section>
      <section>
        {Object.keys(schema).length === 0 && <div className="muted">This pipeline has no params.</div>}
        {groupParams(schema).map(([group, paths]) => {
          const rows = paths.flatMap((path) => Object.entries(schema[path]).filter(([name]) => visible(path, name)).map(([name, info]) => ({ path, name, info })));
          if (!rows.length) return null;
          const count = paths.reduce((n, p) => n + Object.keys(schema[p]).length, 0);
          const edited = rows.filter((r) => isEdited(r.path, r.name)).length;
          const isOpen = open.has(group) || ws.length > 0 || changedOnly;
          const flip = () => {
            const next = new Set(open);
            if (next.has(group)) next.delete(group);
            else next.add(group);
            setOpen(next);
          };
          return (
            <div key={group} className="param-group">
              <h5 className="group" onClick={flip} title={group}>
                {isOpen ? "▾" : "▸"} {group === "shared" ? "Shared tables and rates" : group.split("/").slice(1).join(" / ") || group}{" "}
                <span className="muted small">{ws.length ? `${rows.length} of ${count}` : count} params{edited ? ` · ${edited} changed` : ""}</span>
              </h5>
              {isOpen && (
                <table>
                  <tbody>{rows.map((r) => paramRow(r.path, r.name, r.info))}</tbody>
                </table>
              )}
            </div>
          );
        })}
      </section>
      <section>
        <h4>Change inputs too</h4>
        {overrides.map((o, i) => (
          <div className="override" key={i}>
            <select aria-label="override column" value={o.column} onChange={(e) => setOverrides(overrides.map((x, j) => (j === i ? { ...x, column: e.target.value } : x)))}>
              <option value="">input field…</option>
              {inputColumns.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <input aria-label="override value" placeholder="new value" value={o.value} onChange={(e) => setOverrides(overrides.map((x, j) => (j === i ? { ...x, value: e.target.value } : x)))} />
            <button className="link" onClick={() => setOverrides(overrides.filter((_, j) => j !== i))}>remove</button>
          </div>
        ))}
        <button className="link" onClick={() => setOverrides([...overrides, { column: "", value: "" }])}>+ change an input field</button>
        {overrides.length > 0 && (
          <div className="scope">
            <span className="muted">for</span>
            <label><input type="radio" checked={scope === "all" || record === null} onChange={() => setScope("all")} /> every record</label>
            {record !== null && <label><input type="radio" checked={scope === "record"} onChange={() => setScope("record")} /> {recordLabel(record, keyCol)} only</label>}
          </div>
        )}
      </section>
      <section className="actions sticky">
        <button className="primary" disabled={!changes.length} onClick={() => onWhatIf(doc, override, row, `What-if: ${changes.join("; ")}`)}>
          Run and compare
        </button>
        {sessionRunning && (
          <button disabled={!Object.keys(doc).length} onClick={() => onRestart(doc)}>Restart the debug run with these params</button>
        )}
        <span className="muted small">{changes.length ? `Changing ${changes.join("; ")}` : "Change a value, then run both and compare."}</span>
      </section>
    </div>
  );
}
