import { useEffect, useState } from "react";
import { same } from "../src/compare";
import { formatValue, recordLabel, type ParamInfo, type RecordKey } from "../src/protocol";
import { scenarios, type Knob, type Scenario, type Sweep } from "../src/sweep";
import { parseValue } from "./Params";

const MAX_SCENARIOS = 64;

interface Props {
  schema: Record<string, Record<string, ParamInfo>>;
  /** Names an input knob can set: the state's columns when paused, the input columns otherwise. */
  columns: string[];
  /** The checkpoint scenarios fork from, when a session is paused. */
  pausedAt: string | null;
  record: number | null;
  keyCol: RecordKey;
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

/** Many what-ifs at once: every combination of the knobs, forked from the pause, next to the original. */
export function Scenarios({ schema, columns, pausedAt, record, keyCol, rows, result, onRun, onOpen }: Props) {
  const [knobs, setKnobs] = useState<Row[]>([{ kind: "param", key: "", values: "" }]);
  const [only, setOnly] = useState<number | null>(null);
  const [fromHere, setFromHere] = useState(true);
  const [editing, setEditing] = useState(true);
  const [shownRow, setShownRow] = useState<number>(record ?? 0);
  useEffect(() => {
    if (result.sweep) setEditing(false);
  }, [result.sweep]);
  useEffect(() => {
    if (record !== null) setShownRow(record);
  }, [record]);

  const paramKeys = Object.entries(schema).flatMap(([path, ps]) => Object.keys(ps).map((n) => `${path}|${n}`));
  const built = knobs.map((k) => knobOf(k, schema)).filter((k): k is Knob => k !== null);
  const list = scenarios(built, only);
  const tooMany = list.length > MAX_SCENARIOS;
  const set = (i: number, patch: Partial<Row>) => setKnobs(knobs.map((k, j) => (j === i ? { ...k, ...patch } : k)));
  const start = fromHere && pausedAt ? `from the pause (${pausedAt})` : "from the start";

  if (!editing && result.sweep) {
    return (
      <div className="scenarios">
        <div className="form-summary">
          <span>
            <strong>{result.sweep.labels.length} scenarios</strong> <span className="muted">· {built.map((k) => k.key.split("|").pop()).join(" × ")} · {result.sweep.at ? `forked ${result.sweep.at}` : "from the start"}</span>
          </span>
          <button onClick={() => setEditing(true)}>Edit scenarios</button>
        </div>
        <Results sweep={result.sweep} row={shownRow} rows={rows || result.sweep.rows} onRow={setShownRow} onOpen={onOpen} />
      </div>
    );
  }

  return (
    <div className="scenarios">
      <section>
        <h4>Try every combination of</h4>
        {knobs.map((k, i) => (
          <div className="knob" key={i}>
            <select aria-label="knob kind" value={k.kind} onChange={(e) => set(i, { kind: e.target.value as Row["kind"], key: "" })}>
              <option value="param">parameter</option>
              <option value="value">input field</option>
            </select>
            <select aria-label="knob" value={k.key} onChange={(e) => set(i, { key: e.target.value })}>
              <option value="">{k.kind === "param" ? "choose a parameter…" : "choose a field…"}</option>
              {(k.kind === "param" ? paramKeys : columns).map((key) => (
                <option key={key} value={key}>{key.replace("|", " · ")}</option>
              ))}
            </select>
            <input aria-label="knob values" placeholder="values to try, e.g. 24, 36, 48" value={k.values} onChange={(e) => set(i, { values: e.target.value })} />
            {knobs.length > 1 && <button className="link" onClick={() => setKnobs(knobs.filter((_, j) => j !== i))}>remove</button>}
          </div>
        ))}
        <button className="link" onClick={() => setKnobs([...knobs, { kind: "value", key: "", values: "" }])}>+ add another</button>
        {built.some((k) => k.kind === "value") && (
          <div className="scope">
            <span className="muted">Input fields change for</span>
            <select aria-label="scenario scope" value={only ?? ""} onChange={(e) => setOnly(e.target.value === "" ? null : Number(e.target.value))}>
              <option value="">every record</option>
              {Array.from({ length: rows }, (_, i) => (
                <option key={i} value={i}>{recordLabel(i, keyCol)} only</option>
              ))}
            </select>
          </div>
        )}
        <div className="scope">
          <span className="muted">Run</span>
          <label><input type="radio" checked={fromHere && !!pausedAt} disabled={!pausedAt} onChange={() => setFromHere(true)} /> from the pause{pausedAt ? ` (${pausedAt})` : ""}</label>
          <label><input type="radio" checked={!fromHere || !pausedAt} onChange={() => setFromHere(false)} /> from the start</label>
        </div>
        {fromHere && pausedAt && <div className="muted small">Parameters only change steps after the pause. Input fields replace the values there now.</div>}
      </section>
      <div className="actions sticky">
        <button className="primary" disabled={!list.length || tooMany || !!result.busy} onClick={() => onRun(list, fromHere && !!pausedAt)}>
          {result.busy ? "Running…" : list.length ? `Run ${list.length} scenario${list.length === 1 ? "" : "s"} ${start}` : "Choose a parameter and values to try"}
        </button>
        {tooMany && <span className="error">{list.length} combinations is too many; keep it to {MAX_SCENARIOS}.</span>}
        {result.sweep && <button onClick={() => setEditing(false)}>Back to results</button>}
      </div>
      {result.busy && <div className="empty">{result.busy}</div>}
      {result.error && <div className="empty error">{result.error}</div>}
    </div>
  );
}

/** Up to this many records show side by side; more switch to one record at a time. */
const SIDE_BY_SIDE = 4;

function Results({ sweep, row, rows, onRow, onOpen }: { sweep: Sweep; row: number; rows: number; onRow: (r: number) => void; onOpen: (i: number) => void }) {
  const cols = sweep.changedColumns;
  const knobCols = sweep.knobs.length ? sweep.knobs : [{ name: "scenario", values: sweep.labels }];
  const records = rows <= SIDE_BY_SIDE ? Array.from({ length: rows }, (_, i) => i) : row < 0 ? [] : [row];
  const counting = records.length === 0;
  const recordsOf = (list: number[]) => list.map((r) => recordLabel(r, sweep.key)).join(", ");
  const baseKnob = (name: string) => {
    const values = sweep.knobBase[name] ?? [];
    return values.every((v) => same(v, values[0])) ? formatValue(values[0]) : records.length === 1 ? formatValue(values[records[0]]) : "varies by record";
  };
  const cell = (i: number | null, c: string, r: number) => {
    const before = sweep.base?.[c]?.[r];
    if (i === null) return <td key={`${c}-${r}`} className="mono">{formatValue(before)}</td>;
    const after = sweep.outputs[i]?.[c]?.[r];
    const changed = !same(before, after);
    const delta = changed && typeof before === "number" && typeof after === "number" ? after - before : null;
    return (
      <td key={`${c}-${r}`} className={`mono ${changed ? "changed" : "unchanged"}`} title={changed ? `original run: ${formatValue(before)}` : "same as the original run"}>
        {formatValue(after)}
        {delta !== null && <span className={delta > 0 ? "up" : "down"}> ({delta > 0 ? "+" : ""}{formatValue(delta)})</span>}
      </td>
    );
  };
  return (
    <section>
      {rows > SIDE_BY_SIDE && (
        <div className="summary">
          <span>Results for</span>
          <select aria-label="scenario record" value={row} onChange={(e) => onRow(Number(e.target.value))}>
            <option value={-1}>all records (count changes)</option>
            {Array.from({ length: rows }, (_, i) => (
              <option key={i} value={i}>{recordLabel(i, sweep.key)}</option>
            ))}
          </select>
        </div>
      )}
      <p className="hint">One row per scenario. Highlighted cells differ from the original run. Click a row for its step-by-step comparison.</p>
      {cols.length === 0 ? (
        <div className="muted">No scenario changes any result.</div>
      ) : (
        <table className="sweep">
          <thead>
            {records.length > 1 && (
              <tr>
                <th colSpan={knobCols.length} className="knob-col">tried</th>
                {records.map((r) => (
                  <th key={r} colSpan={cols.length} className="group-head">{recordLabel(r, sweep.key)}</th>
                ))}
                <th />
              </tr>
            )}
            <tr>
              {knobCols.map((k) => (
                <th key={k.name} className="knob-col" title={k.name}>{k.name.split(" · ").pop()}</th>
              ))}
              {(counting ? [0] : records).flatMap((r) => cols.map((c) => <th key={`${c}-${r}`}>{c}</th>))}
              <th title="How many records have any result that differs from the original run">records changed</th>
            </tr>
          </thead>
          <tbody>
            <tr className="original">
              {knobCols.map((k) => (
                <td key={k.name} className="mono knob-col">{k.name === "scenario" ? "original run" : baseKnob(k.name)}</td>
              ))}
              {counting ? cols.map((c) => <td key={c} />) : records.flatMap((r) => cols.map((c) => cell(null, c, r)))}
              <td className="muted">original run</td>
            </tr>
            {sweep.labels.map((label, i) => {
              const changedRows = [...new Set(sweep.comparisons[i].output.flatMap((o) => o.changedRows))].sort((a, b) => a - b);
              return (
                <tr key={i} className="clickable" title={`${label}: open its step-by-step comparison`} onClick={() => onOpen(i)}>
                  {knobCols.map((k) => (
                    <td key={k.name} className="mono knob-col">{formatValue(k.values[i])}</td>
                  ))}
                  {counting
                    ? cols.map((c) => {
                        const diff = sweep.comparisons[i].output.find((o) => o.name === c);
                        return (
                          <td key={c} className={diff ? "changed" : "unchanged"} title={diff ? `changed for ${recordsOf(diff.changedRows)}` : "same for every record"}>
                            {diff ? `${diff.changedRows.length} changed` : "same"}
                          </td>
                        );
                      })
                    : records.flatMap((r) => cols.map((c) => cell(i, c, r)))}
                  <td title={changedRows.length ? recordsOf(changedRows) : "no record changed"}>
                    {sweep.errors[i] ? <span className="error small">{sweep.errors[i]}</span> : `${changedRows.length} of ${rows}`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}
