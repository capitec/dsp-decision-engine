import { useEffect, useState } from "react";
import { same } from "../src/compare";
import { formatValue, recordLabel, type ParamInfo, type RecordKey } from "../src/protocol";
import { scenarios, type Knob, type Scenario, type Sweep } from "../src/sweep";
import { Compare } from "./Compare";
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
  onSelectStep: (path: string) => void;
  onCompareRevision: () => void;
  onOpenDiff: (path: string) => void;
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
export function Scenarios({ schema, columns, pausedAt, record, keyCol, rows, result, onRun, onOpen, onSelectStep, onCompareRevision, onOpenDiff }: Props) {
  const [knobs, setKnobs] = useState<Row[]>([{ kind: "param", key: "", values: "" }]);
  const [only, setOnly] = useState<number | null>(null);
  const [fromHere, setFromHere] = useState(true);
  const [editing, setEditing] = useState(true);
  // A few records fit side by side; with more, a summary per scenario.
  const [shownRow, setShownRow] = useState<number>(record ?? (rows <= 3 ? -2 : -1));
  const [open, setOpen] = useState<number | null>(null);
  useEffect(() => {
    if (result.sweep) setEditing(false);
    setOpen(null);
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
        <Results
          sweep={result.sweep}
          row={shownRow}
          rows={rows || result.sweep.rows}
          open={open}
          onRow={setShownRow}
          onOpen={(i) => {
            setOpen(open === i ? null : i);
            onOpen(i);
          }}
        />
        {open !== null && (
          <div className="inline-compare" ref={(el) => el?.scrollIntoView({ block: "start", behavior: "smooth" })}>
            <Compare
              withRevision={false}
              header={
                <div className="pager">
                  <button title="Previous scenario" onClick={() => { const i = (open + result.sweep!.labels.length - 1) % result.sweep!.labels.length; setOpen(i); onOpen(i); }}>◀</button>
                  <strong>Scenario {open + 1} of {result.sweep.labels.length}</strong>
                  <button title="Next scenario" onClick={() => { const i = (open + 1) % result.sweep!.labels.length; setOpen(i); onOpen(i); }}>▶</button>
                  <a onClick={() => setOpen(null)}>back to all scenarios</a>
                </div>
              }
              comparison={result.sweep.comparisons[open]}
              record={shownRow < 0 ? null : shownRow}
              onSelect={onSelectStep}
              onCompareRevision={onCompareRevision}
              onOpenDiff={onOpenDiff}
            />
          </div>
        )}
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
                <option key={key} value={key} title={key.replace("|", " · ")}>{k.kind === "param" ? `${key.split("|")[0].split("/").pop()} · ${key.split("|")[1]}` : key}</option>
              ))}
            </select>
            <input aria-label="knob values" className="grow" placeholder="values to try, e.g. 24, 36, 48" value={k.values} onChange={(e) => set(i, { values: e.target.value })} />
            <button className="link" style={{ visibility: knobs.length > 1 ? "visible" : "hidden" }} onClick={() => setKnobs(knobs.filter((_, j) => j !== i))}>remove</button>
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
          <label><input type="radio" checked={fromHere && !!pausedAt} disabled={!pausedAt} onChange={() => setFromHere(true)} /> from where the run is paused</label>
          <label><input type="radio" checked={!fromHere || !pausedAt} onChange={() => setFromHere(false)} /> from the start</label>
        </div>
        {fromHere && pausedAt && (
          <div className="muted">
            Starts {pausedAt}: changed parameters affect that step and the ones after it; changed input fields overwrite their current values.
          </div>
        )}
      </section>
      <div className="actions sticky">
        <button className="primary" disabled={!list.length || tooMany || !!result.busy} onClick={() => onRun(list, fromHere && !!pausedAt)}>
          {result.busy ? "Running…" : list.length ? `Run ${list.length} scenario${list.length === 1 ? "" : "s"}` : "Run"}
        </button>
        <span>{list.length ? start : "Pick a parameter or input field and the values to try."}</span>
        {tooMany && <span className="error">{list.length} combinations is too many; keep it to {MAX_SCENARIOS}.</span>}
        {result.sweep && <button onClick={() => setEditing(false)}>Back to results</button>}
      </div>
      {result.busy && <div className="empty">{result.busy}</div>}
      {result.error && <div className="empty error">{result.error}</div>}
    </div>
  );
}

function Results({ sweep, row, rows, onRow, onOpen, open }: { sweep: Sweep; row: number; rows: number; onRow: (r: number) => void; onOpen: (i: number) => void; open: number | null }) {
  // The outcomes that moved most come first, so the columns that matter survive a narrow panel.
  const moved = (c: string) => sweep.comparisons.reduce((t, cmp) => t + (cmp.output.find((o) => o.name === c)?.changedRows.length ?? 0), 0);
  const cols = [...sweep.changedColumns].sort((a, b) => moved(b) - moved(a));
  const knobCols = sweep.knobs.length ? sweep.knobs : [{ name: "scenario", values: sweep.labels }];
  const summary = row === -1;
  const each = row === -2;
  const shown = each ? Array.from({ length: rows }, (_, i) => i) : [row];
  const recordsOf = (list: number[]) => list.map((r) => recordLabel(r, sweep.key)).join(", ");
  const baseKnob = (name: string) => {
    const values = sweep.knobBase[name] ?? [];
    if (!summary && !each) return formatValue(values[row]);
    return values.every((v) => same(v, values[0])) ? formatValue(values[0]) : values.map(formatValue).join(" / ");
  };
  // The mean over all records, or the single value when every record agrees.
  const overall = (values: unknown[] | undefined) => {
    const v = values ?? [];
    if (v.every((x) => same(x, v[0]))) return formatValue(v[0]);
    const nums = v.filter((x): x is number => typeof x === "number");
    return nums.length === v.length ? `${formatValue(Math.min(...nums))}–${formatValue(Math.max(...nums))}` : "varies";
  };
  const summaryCell = (i: number, c: string) => {
    const diff = sweep.comparisons[i].output.find((o) => o.name === c);
    const value = overall(sweep.outputs[i]?.[c]);
    if (!diff) return <td key={c} className="unchanged mono" title="same as the original run for every record">{value}</td>;
    return (
      <td key={c} className="changed mono" title={`changed for ${recordsOf(diff.changedRows)}; original: ${overall(sweep.base?.[c])}`}>
        {value}
        <span className="up"> ({diff.changedRows.length}/{rows} changed)</span>
      </td>
    );
  };
  const recordCell = (i: number | null, c: string, r: number) => {
    const before = sweep.base?.[c]?.[r];
    if (i === null) return <td key={`${c}-${r}`} className="mono">{formatValue(before)}</td>;
    const after = sweep.outputs[i]?.[c]?.[r];
    const changed = !same(before, after);
    return (
      <td key={`${c}-${r}`} className={`mono ${changed ? "changed" : "unchanged"}`} title={changed ? `original run: ${formatValue(before)}` : "same as the original run"}>
        {changed ? <><s className="before">{formatValue(before)}</s> {formatValue(after)}</> : `= ${formatValue(after)}`}
      </td>
    );
  };
  return (
    <section>
      <div className="summary">
        <span>Show</span>
        <select aria-label="scenario record" value={row} onChange={(e) => onRow(Number(e.target.value))}>
          <option value={-2}>each record side by side</option>
          <option value={-1}>a summary of all {rows} records (ranges)</option>
          {Array.from({ length: rows }, (_, i) => (
            <option key={i} value={i}>the values for {recordLabel(i, sweep.key)}</option>
          ))}
        </select>
      </div>
      <p className="hint">One row per scenario; highlighted cells differ from the original run. Click a row to see what changed, step by step.</p>
      {cols.length === 0 ? (
        <div className="muted">No scenario changes any result.</div>
      ) : (
        <table className="sweep">
          <thead>
            {each && (
              <tr>
                <th colSpan={knobCols.length} />
                {shown.map((r) => (
                  <th key={r} colSpan={cols.length} className="group-head">{recordLabel(r, sweep.key)}</th>
                ))}
              </tr>
            )}
            <tr>
              {knobCols.map((k) => (
                <th key={k.name} className="knob-col" title={k.name}>{k.name.split(" · ").pop()}</th>
              ))}
              {(summary ? [0] : shown).flatMap((r) => cols.map((c) => <th key={`${c}-${r}`}>{c}</th>))}
            </tr>
          </thead>
          <tbody>
            <tr className="original">
              <td className="knob-col" colSpan={knobCols.length} title={knobCols.map((k) => `${k.name} = ${baseKnob(k.name)}`).join("\n")}>
                original run ({knobCols.filter((k) => k.name !== "scenario").map((k) => `${k.name.split(" · ").pop()} ${baseKnob(k.name)}`).join(", ")})
              </td>
              {summary ? cols.map((c) => <td key={c} className="mono">{overall(sweep.base?.[c])}</td>) : shown.flatMap((r) => cols.map((c) => recordCell(null, c, r)))}
            </tr>
            {sweep.labels.map((label, i) => (
              <tr key={i} className={`clickable ${open === i ? "open" : ""}`} title={`${label}: see what changed, step by step`} onClick={() => onOpen(i)}>
                {knobCols.map((k) => (
                  <td key={k.name} className="mono knob-col">{formatValue(k.values[i])}</td>
                ))}
                {summary ? cols.map((c) => summaryCell(i, c)) : shown.flatMap((r) => cols.map((c) => recordCell(i, c, r)))}
                {sweep.errors[i] && <td className="error small">{sweep.errors[i]}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
