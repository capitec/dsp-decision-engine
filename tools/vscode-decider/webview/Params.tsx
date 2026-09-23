import { useState } from "react";
import type { ParamInfo } from "../src/protocol";

interface Props {
  schema: Record<string, Record<string, ParamInfo>>;
  inputColumns: string[];
  record: number | null;
  sessionRunning: boolean;
  onWhatIf: (params: unknown, overrides: Record<string, unknown>, row: number | null) => void;
  onRestart: (params: unknown) => void;
  onCompareRevision: () => void;
}

type Edits = Record<string, string>;

/** A params document holding only the edited values, nested by path. */
export function paramsDocument(schema: Props["schema"], edits: Edits): Record<string, unknown> {
  const doc: Record<string, unknown> = {};
  for (const [key, text] of Object.entries(edits)) {
    if (text.trim() === "") continue;
    const [path, name] = key.split("|");
    let target = doc as Record<string, Record<string, unknown>>;
    for (const part of path.split("/")) target = (target[part] ??= {}) as Record<string, Record<string, unknown>>;
    (target as Record<string, unknown>)[name] = parseValue(text, schema[path]?.[name]?.type);
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

export function Params({ schema, inputColumns, record, sessionRunning, onWhatIf, onRestart, onCompareRevision }: Props) {
  const [edits, setEdits] = useState<Edits>({});
  const [overrides, setOverrides] = useState<{ column: string; value: string }[]>([]);
  const [scope, setScope] = useState<"record" | "all">("record");
  const doc = paramsDocument(schema, edits);
  const override = Object.fromEntries(overrides.filter((o) => o.column && o.value !== "").map((o) => [o.column, parseValue(o.value, "number")]));
  const row = scope === "record" && record !== null ? record : null;
  const changed = Object.keys(doc).length + Object.keys(override).length;

  return (
    <div className="params">
      <section>
        <h4>Params</h4>
        {Object.keys(schema).length === 0 && <div className="muted">This pipeline has no params.</div>}
        <table>
          <tbody>
            {Object.entries(schema).map(([path, params]) => [
              <tr key={path} className="group">
                <td colSpan={4}>{path}</td>
              </tr>,
              ...Object.entries(params).map(([name, info]) => {
                const key = `${path}|${name}`;
                const edited = (edits[key] ?? "").trim() !== "";
                return (
                  <tr key={key} className={edited ? "edited" : ""}>
                    <td className="name">{name}</td>
                    <td className="muted">{info.type}{bounds(info) && ` · ${bounds(info)}`}</td>
                    <td>
                      <input
                        aria-label={`${path} ${name}`}
                        placeholder={JSON.stringify(info.default ?? null)}
                        value={edits[key] ?? ""}
                        onChange={(e) => setEdits({ ...edits, [key]: e.target.value })}
                      />
                    </td>
                    <td>{edited && <button className="link" onClick={() => setEdits({ ...edits, [key]: "" })}>reset</button>}</td>
                  </tr>
                );
              }),
            ])}
          </tbody>
        </table>
      </section>
      <section>
        <h4>Input overrides</h4>
        {overrides.map((o, i) => (
          <div className="override" key={i}>
            <select aria-label="override column" value={o.column} onChange={(e) => setOverrides(overrides.map((x, j) => (j === i ? { ...x, column: e.target.value } : x)))}>
              <option value="">column…</option>
              {inputColumns.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <input aria-label="override value" placeholder="value" value={o.value} onChange={(e) => setOverrides(overrides.map((x, j) => (j === i ? { ...x, value: e.target.value } : x)))} />
            <button className="link" onClick={() => setOverrides(overrides.filter((_, j) => j !== i))}>remove</button>
          </div>
        ))}
        <button className="link" onClick={() => setOverrides([...overrides, { column: "", value: "" }])}>+ add an input override</button>
        {overrides.length > 0 && (
          <div className="scope">
            <label><input type="radio" checked={scope === "record"} disabled={record === null} onChange={() => setScope("record")} /> record {record ?? "(focus one in a session)"}</label>
            <label><input type="radio" checked={scope === "all" || record === null} onChange={() => setScope("all")} /> every record</label>
          </div>
        )}
      </section>
      <section className="actions">
        <button className="primary" disabled={!changed} onClick={() => onWhatIf(doc, override, row)}>Compare what-if with defaults</button>
        <button disabled={!sessionRunning || !Object.keys(doc).length} onClick={() => onRestart(doc)} title="Restart the debug session with these params">Restart session with these params</button>
        <button onClick={onCompareRevision}>Compare with a git revision…</button>
      </section>
    </div>
  );
}
