import { useEffect, useState } from "react";
import { formatValue, isRateName, recordLabel, type ParamInfo, type RecordKey } from "./model/protocol";
import { scenarios, type Knob, type Scenario, type Sweep } from "./model/sweep";
import { Compare } from "./Compare";
import { SweepResults } from "./SweepResults";
import { currentValue, parseValue } from "./Params";

const MAX_SCENARIOS = 64;

interface Props {
  schema: Record<string, Record<string, ParamInfo>>;
  /** The flow's PARAMS document, to show what a param is now. */
  values: Record<string, unknown>;
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
  onCompareRevision?: () => void;
  onOpenDiff?: (path: string) => void;
}

interface Row {
  kind: "param" | "value";
  key: string;
  /** What is typed in the picker; `key` is set once it names a param or field. */
  text?: string;
  values: string;
}

/** "cap · pl_product_cap (personal_loan/limits)": a param's name, its step and where the step sits. */
export function paramLabel(key: string): string {
  const [path, name] = key.split("|");
  if (path === "shared") return `${name} (shared)`;
  const parts = path.split("/");
  const group = parts.slice(0, -1).slice(-2).join("/");
  const step = parts[parts.length - 1];
  return `${step.endsWith(name) ? step : `${step} (${name})`}${group ? ` in ${group}` : ""}`;
}

function knobOf(r: Row, schema: Props["schema"]): Knob | null {
  const parts = r.values.split(",").map((v) => v.trim()).filter(Boolean);
  if (!r.key || !parts.length) return null;
  const [path, name] = r.key.split("|");
  const type = r.kind === "param" ? schema[path]?.[name]?.type : "number";
  return { kind: r.kind, key: r.key, values: parts.map((v) => parseValue(v, type)) };
}

/** Many what-ifs at once: every combination of the knobs, forked from the pause, next to the original. */
export function Scenarios({ schema, values, columns, pausedAt, record, keyCol, rows, result, onRun, onOpen, onSelectStep, onCompareRevision, onOpenDiff }: Props) {
  const [knobs, setKnobs] = useState<Row[]>([{ kind: "param", key: "", values: "" }]);
  const [only, setOnly] = useState<number | null>(null);
  const [fromHere, setFromHere] = useState(true);
  const [editing, setEditing] = useState(true);
  // A few records fit side by side; with more, a summary per scenario.
  const [shownRow, setShownRow] = useState<number>(record ?? (rows <= 3 ? -2 : -1));
  const [open, setOpen] = useState<number | null>(null);
  useEffect(() => {
    if (result.sweep) {
      setEditing(false);
      if (record === null) setShownRow(result.sweep.rows <= 3 ? -2 : -1);
    }
    setOpen(null);
  }, [result.sweep]);
  useEffect(() => {
    if (record !== null) setShownRow(record);
  }, [record]);

  const currentOf = (key: string) => {
    const [path, name] = key.split("|");
    return currentValue(values, path, name, schema[path][name]);
  };
  const paramKeys = Object.entries(schema).flatMap(([path, ps]) => Object.keys(ps).filter((n) => ps[n].type !== "table").map((n) => `${path}|${n}`));
  const labels = new Map(paramKeys.map((k) => [paramLabel(k), k]));
  const incomplete = knobs.map((k) => (k.text || k.key || k.values.trim() ? !knobOf(k, schema) : false));
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
        <SweepResults
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
          <div className="inline-compare" ref={(el) => void el?.scrollIntoView({ block: "start", behavior: "smooth" })}>
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
            <input
              aria-label="knob"
              list="knob-options"
              className={`picker ${incomplete[i] && !k.key ? "invalid" : ""}`}
              placeholder="type a parameter or an input field…"
              title={k.key.replace("|", " · ")}
              value={k.text ?? (k.kind === "param" && k.key ? paramLabel(k.key) : k.key ? `field · ${k.key}` : "")}
              onChange={(e) => {
                const text = e.target.value;
                const field = text.startsWith("field · ") ? text.slice("field · ".length) : "";
                if (labels.has(text)) set(i, { text, kind: "param", key: labels.get(text)! });
                else set(i, { text, kind: "value", key: columns.includes(field) ? field : "" });
              }}
            />
            <input
              aria-label="knob values"
              className={`values ${incomplete[i] && k.key ? "invalid" : ""}`}
              placeholder={k.key && isRateName(k.key.split("|").pop()) ? "values, e.g. 7%, 7.75%, 8.5%" : "values, e.g. 24, 36, 48"}
              value={k.values}
              onChange={(e) => set(i, { values: e.target.value })}
            />
            <button className="link" style={{ visibility: knobs.length > 1 ? "visible" : "hidden" }} onClick={() => setKnobs(knobs.filter((_, j) => j !== i))}>remove</button>
            {(k.key || knobOf(k, schema)) && (
              <div className="muted small knob-echo" title={k.key.replace("|", " · ")}>
                {k.kind === "param" && k.key ? `now ${formatValue(currentOf(k.key), k.key.split("|")[1])}` : ""}
                {knobOf(k, schema) ? `${k.kind === "param" && k.key ? " · " : ""}will try ${knobOf(k, schema)!.values.map((v) => formatValue(v, k.key.split("|").pop())).join(", ")}` : ""}
              </div>
            )}
          </div>
        ))}
        <datalist id="knob-options">
          {[...labels.keys(), ...columns.map((c) => `field · ${c}`)].map((label) => (
            <option key={label} value={label} />
          ))}
        </datalist>
        <button className="link" onClick={() => setKnobs([...knobs, { kind: "param", key: "", values: "" }])}>+ add another</button>
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
        {pausedAt && (
          <div className="scope">
            <span className="muted">Run</span>
            <label><input type="radio" checked={fromHere} onChange={() => setFromHere(true)} /> from where the run is paused</label>
            <label><input type="radio" checked={!fromHere} onChange={() => setFromHere(false)} /> from the start</label>
          </div>
        )}
        {fromHere && pausedAt && (
          <div className="muted">
            Starts {pausedAt}: changed parameters affect that step and the ones after it; changed input fields overwrite their current values.
          </div>
        )}
      </section>
      <div className="actions sticky">
        <button className="primary" disabled={!list.length || tooMany || incomplete.some(Boolean) || !!result.busy} onClick={() => onRun(list, fromHere && !!pausedAt)}>
          {result.busy ? "Running…" : list.length ? `Run ${list.length} scenario${list.length === 1 ? "" : "s"}` : "Run"}
        </button>
        <span>{list.length ? start : "Pick a parameter or input field and the values to try."}</span>
        {tooMany && <span className="error">{list.length} combinations is too many; keep it to {MAX_SCENARIOS}.</span>}
        {incomplete.some(Boolean) && <span className="error">Finish or remove the row marked red: it needs {incomplete.findIndex(Boolean) >= 0 && !knobs[incomplete.findIndex(Boolean)].key ? "a parameter or field from the list" : "values to try"}.</span>}
        {result.sweep && <button onClick={() => setEditing(false)}>Back to results</button>}
      </div>
      {result.busy && <div className="empty">{result.busy}</div>}
      {result.error && <div className="empty error">{result.error}</div>}
    </div>
  );
}
