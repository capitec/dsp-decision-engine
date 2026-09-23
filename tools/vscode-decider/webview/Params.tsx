import { useState } from "react";
import { formatValue, recordLabel, type ParamInfo, type RecordKey } from "../src/protocol";

interface Props {
  schema: Record<string, Record<string, ParamInfo>>;
  inputColumns: string[];
  record: number | null;
  keyCol: RecordKey;
  sessionRunning: boolean;
  onWhatIf: (params: unknown, overrides: Record<string, unknown>, row: number | null, label: string) => void;
  onRestart: (params: unknown) => void;
}

type Edits = Record<string, string>;

const shownDefault = (info: ParamInfo) => formatValue(info.default ?? null);

/** A params document holding only the values that differ from their defaults, nested by path. */
export function paramsDocument(schema: Props["schema"], edits: Edits): Record<string, unknown> {
  const doc: Record<string, unknown> = {};
  for (const [key, text] of Object.entries(edits)) {
    const [path, name] = key.split("|");
    const info = schema[path]?.[name];
    if (text.trim() === "" || !info || text.trim() === shownDefault(info)) continue;
    let target = doc as Record<string, Record<string, unknown>>;
    for (const part of path.split("/")) target = (target[part] ??= {}) as Record<string, Record<string, unknown>>;
    (target as Record<string, unknown>)[name] = parseValue(text, info.type);
  }
  return doc;
}

export function parseValue(text: string, type?: string): unknown {
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

function leafNames(doc: Record<string, unknown>, prefix = ""): string[] {
  return Object.entries(doc).flatMap(([k, v]) =>
    v && typeof v === "object" && !Array.isArray(v) ? leafNames(v as Record<string, unknown>, prefix ? `${prefix}/${k}` : k) : [`${prefix} · ${k} = ${formatValue(v)}`],
  );
}

/** Every param with its default; change some, then compare with the defaults or restart the run with them. */
export function Params({ schema, inputColumns, record, keyCol, sessionRunning, onWhatIf, onRestart }: Props) {
  const [edits, setEdits] = useState<Edits>({});
  const [overrides, setOverrides] = useState<{ column: string; value: string }[]>([]);
  const [scope, setScope] = useState<"record" | "all">("record");
  const doc = paramsDocument(schema, edits);
  const override = Object.fromEntries(overrides.filter((o) => o.column && o.value !== "").map((o) => [o.column, parseValue(o.value, "number")]));
  const row = scope === "record" && record !== null ? record : null;
  const who = row === null ? "every record" : recordLabel(row, keyCol);
  const changes = [...leafNames(doc), ...Object.entries(override).map(([k, v]) => `${k} = ${formatValue(v)} for ${who}`)];

  return (
    <div className="params">
      <section>
        <h4>Params</h4>
        {Object.keys(schema).length === 0 && <div className="muted">This pipeline has no params.</div>}
        <table>
          <tbody>
            {Object.entries(schema).map(([path, params]) => [
              <tr key={path} className="group" title={path}>
                <td colSpan={3}>{path.split("/").pop()} <span className="muted">{path.includes("/") ? `in ${path.slice(0, path.lastIndexOf("/"))}` : ""}</span></td>
              </tr>,
              ...Object.entries(params).map(([name, info]) => {
                const key = `${path}|${name}`;
                const text = edits[key] ?? shownDefault(info);
                const edited = text.trim() !== shownDefault(info);
                return (
                  <tr key={key} className={edited ? "edited" : ""}>
                    <td className="name">{name}</td>
                    <td>
                      <input aria-label={`${path} ${name}`} value={text} onChange={(e) => setEdits({ ...edits, [key]: e.target.value })} />
                    </td>
                    {bounds(info) && <td className="muted" title={info.type}>{bounds(info)}</td>}
                    <td>
                      {edited ? (
                        <button className="link" onClick={() => setEdits({ ...edits, [key]: shownDefault(info) })}>reset to {shownDefault(info)}</button>
                      ) : (
                        <span className="muted small">default</span>
                      )}
                    </td>
                  </tr>
                );
              }),
            ])}
          </tbody>
        </table>
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
        <button className="primary" disabled={!changes.length} onClick={() => onWhatIf(doc, override, row, changes.join(", "))}>
          Compare with defaults
        </button>
        {sessionRunning && (
          <button disabled={!Object.keys(doc).length} onClick={() => onRestart(doc)}>Restart the debug run with these params</button>
        )}
        <span className="muted small">{changes.length ? `Changing: ${changes.join(", ")}` : "Change a value to compare."}</span>
      </section>
    </div>
  );
}
