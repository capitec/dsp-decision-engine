import { useState } from "react";
import { same } from "../src/compare";
import type { ParamInfo } from "../src/protocol";
import { scenarios, type Knob, type Scenario, type Sweep } from "../src/sweep";
import { parseValue } from "./Params";

const MAX_SCENARIOS = 64;

interface Props {
  schema: Record<string, Record<string, ParamInfo>>;
  /** Names a value knob can set: the state's columns when paused, the input columns otherwise. */
  columns: string[];
  /** The checkpoint scenarios fork from, when a session is paused. */
  pausedAt: string | null;
  record: number | null;
  rows: number;
  result: { sweep: Sweep | null; busy?: string; error?: string };
  onRun: (scenarios: Scenario[], fromHere: boolean) => void;
  onOpen: (index: number) => void;
}

interface Row {
  kind: "param" | "value";
  key: string;
  values: string;
}

function knobOf(r: Row, schema: Props["schema"]): Knob | null {
  const parts = r.values.split(",").map((v) => v.trim()).filter(Boolean);
  if (!r.key || !parts.length) return null;
  const [path, name] = r.key.split("|");
  const type = r.kind === "param" ? schema[path]?.[name]?.type : "number";
  return { kind: r.kind, key: r.key, values: parts.map((v) => parseValue(v, type)) };
}

/** Many what-ifs at once: every combination of the knobs, forked from the pause, against the original. */
export function Scenarios({ schema, columns, pausedAt, record, rows, result, onRun, onOpen }: Props) {
  const [knobs, setKnobs] = useState<Row[]>([{ kind: "param", key: "", values: "" }]);
  const [scope, setScope] = useState<"record" | "all">("all");
  const [fromHere, setFromHere] = useState(true);
  const [pick, setPick] = useState<number | null>(null);
  const paramKeys = Object.entries(schema).flatMap(([path, ps]) => Object.keys(ps).map((n) => `${path}|${n}`));
  const built = knobs.map((k) => knobOf(k, schema)).filter((k): k is Knob => k !== null);
  const row = scope === "record" ? record : null;
  const list = scenarios(built, row);
  const tooMany = list.length > MAX_SCENARIOS;
  const set = (i: number, patch: Partial<Row>) => setKnobs(knobs.map((k, j) => (j === i ? { ...k, ...patch } : k)));
  const shownRow = pick ?? record ?? 0;

  return (
    <div className="scenarios">
      <section>
        <h4>Try every combination of</h4>
        {knobs.map((k, i) => (
          <div className="knob" key={i}>
            <select aria-label="knob kind" value={k.kind} onChange={(e) => set(i, { kind: e.target.value as Row["kind"], key: "" })}>
              <option value="param">param</option>
              <option value="value">value</option>
            </select>
            <select aria-label="knob" value={k.key} onChange={(e) => set(i, { key: e.target.value })}>
              <option value="">{k.kind === "param" ? "choose a param…" : "choose a column…"}</option>
              {(k.kind === "param" ? paramKeys : columns).map((key) => (
                <option key={key} value={key}>{key.replace("|", " · ")}</option>
              ))}
            </select>
            <input aria-label="knob values" placeholder="values, comma separated: 24, 36, 48" value={k.values} onChange={(e) => set(i, { values: e.target.value })} />
            {knobs.length > 1 && <button className="link" onClick={() => setKnobs(knobs.filter((_, j) => j !== i))}>remove</button>}
          </div>
        ))}
        <button className="link" onClick={() => setKnobs([...knobs, { kind: "value", key: "", values: "" }])}>+ add a knob</button>
        <div className="scope">
          <span className="muted">Values apply to</span>
          <label><input type="radio" checked={scope === "all" || record === null} onChange={() => setScope("all")} /> every record</label>
          <label><input type="radio" checked={scope === "record" && record !== null} disabled={record === null} onChange={() => setScope("record")} /> record {record ?? "(focus one first)"}</label>
        </div>
        <div className="scope">
          <span className="muted">Start from</span>
          <label><input type="radio" checked={fromHere && !!pausedAt} disabled={!pausedAt} onChange={() => setFromHere(true)} /> the pause{pausedAt ? ` (${pausedAt})` : " (run the flow and pause first)"}</label>
          <label><input type="radio" checked={!fromHere || !pausedAt} onChange={() => setFromHere(false)} /> the start</label>
        </div>
        <div className="actions">
          <button className="primary" disabled={!list.length || tooMany || !!result.busy} onClick={() => onRun(list, fromHere && !!pausedAt)}>
            Run {list.length || ""} scenario{list.length === 1 ? "" : "s"}
          </button>
          {tooMany && <span className="error">That is {list.length} combinations; keep it under {MAX_SCENARIOS}.</span>}
          {!fromHere || !pausedAt ? null : <span className="muted">Params only change steps after the pause; values replace what is there now.</span>}
        </div>
      </section>
      {result.busy && <div className="empty">{result.busy}</div>}
      {result.error && <div className="empty error">{result.error}</div>}
      {result.sweep && <Results sweep={result.sweep} row={shownRow} rows={rows || result.sweep.rows} onRow={setPick} onOpen={onOpen} />}
    </div>
  );
}

function Results({ sweep, row, rows, onRow, onOpen }: { sweep: Sweep; row: number; rows: number; onRow: (r: number) => void; onOpen: (i: number) => void }) {
  const cols = sweep.changedColumns;
  const fmt = (v: unknown) => (v === undefined ? "—" : typeof v === "number" ? String(Math.round(v * 1e6) / 1e6) : JSON.stringify(v));
  return (
    <section>
      <div className="summary">
        <strong>{sweep.labels.length} scenarios</strong>
        <span className="muted"> forked {sweep.at ?? "from the start"}</span>
        <label className="right">
          record{" "}
          <select aria-label="scenario record" value={row} onChange={(e) => onRow(Number(e.target.value))}>
            {Array.from({ length: rows }, (_, i) => (
              <option key={i} value={i}>{i}</option>
            ))}
          </select>
        </label>
      </div>
      {cols.length === 0 ? (
        <div className="muted">No scenario changes any output.</div>
      ) : (
        <table className="sweep">
          <thead>
            <tr>
              <th>scenario</th>
              <th title="Steps whose outputs differ from the original, across all records">steps changed</th>
              {cols.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="original">
              <td>original</td>
              <td />
              {cols.map((c) => (
                <td key={c} className="mono">{fmt(sweep.base?.[c]?.[row])}</td>
              ))}
            </tr>
            {sweep.labels.map((label, i) => (
              <tr key={i}>
                <td>
                  <a title="Open the step-by-step comparison" onClick={() => onOpen(i)}>{label}</a>
                  {sweep.errors[i] && <div className="error small">{sweep.errors[i]}</div>}
                </td>
                <td>{sweep.comparisons[i].steps.filter((s) => s.outputs.length).length}</td>
                {cols.map((c) => {
                  const before = sweep.base?.[c]?.[row];
                  const after = sweep.outputs[i]?.[c]?.[row];
                  const changed = !same(before, after);
                  const delta = changed && typeof before === "number" && typeof after === "number" ? after - before : null;
                  return (
                    <td key={c} className={`mono ${changed ? "changed" : "unchanged"}`} title={changed ? `original: ${fmt(before)}` : "same as the original"}>
                      {fmt(after)}
                      {delta !== null && <span className={delta > 0 ? "up" : "down"}> {delta > 0 ? "+" : ""}{fmt(delta)}</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
