import { useEffect, useState } from "react";
import { same } from "../src/compare";
import { formatValue, isRateName, recordLabel, type ParamInfo, type RecordKey } from "../src/protocol";
import { scenarios, type Knob, type Scenario, type Sweep } from "../src/sweep";
import { Compare } from "./Compare";
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
  onCompareRevision: () => void;
  onOpenDiff: (path: string) => void;
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
  return `${parts[parts.length - 1]} (${name})${group ? ` in ${group}` : ""}`;
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
            <button className="link" style={{ visibility: knobs.length > 1 ? "visible" : "hidden" }} onClick={() => setKnobs(knobs.filter((_, j) => j !== i))}>remove</button>
            {knobOf(k, schema) && (
              <div className="muted small knob-echo">tries {knobOf(k, schema)!.values.map((v) => formatValue(v, k.key.split("|").pop())).join(", ")}</div>
            )}
            {k.kind === "param" && k.key && (
              <div className="muted small knob-now" title={k.key.replace("|", " · ")}>
                {k.key.split("|")[0].split("/").slice(-3).join("/")} · now {formatValue(currentOf(k.key), k.key.split("|")[1])}
              </div>
            )}
            <input aria-label="knob values" className={`values ${incomplete[i] && k.key ? "invalid" : ""}`} placeholder={k.key && isRateName(k.key.split("|").pop()) ? "values to try, e.g. 7%, 7.75%, 8.5%" : "values to try, e.g. 24, 36, 48"} value={k.values} onChange={(e) => set(i, { values: e.target.value })} />
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

const categorical = (v: unknown[] | undefined) => !!v?.length && v.every((x) => typeof x === "string" || typeof x === "boolean");

/** "pl_product_cap cap" for a step's param, "repo_rate" for a shared one, the field for an input. */
function knobShort(name: string): string {
  const [path, param] = name.split(" · ");
  if (param === undefined || path === "shared") return param ?? name;
  const step = path.split("/").pop()!;
  return step.includes(param) ? step : `${step} ${param}`;
}

function Results({ sweep, row, rows, onRow, onOpen, open }: { sweep: Sweep; row: number; rows: number; onRow: (r: number) => void; onOpen: (i: number) => void; open: number | null }) {
  // The outcomes that moved most come first, so the columns that matter survive a narrow panel.
  const moved = (c: string) => sweep.comparisons.reduce((t, cmp) => t + (cmp.output.find((o) => o.name === c)?.changedRows.length ?? 0), 0);
  const changedCols = [...sweep.changedColumns].sort((a, b) => moved(b) - moved(a));
  // Outcomes with a few values (approve / refer / decline) always show in the summary, changed or not.
  const outcomes = Object.entries(sweep.base ?? {})
    .filter(([c, v]) => !changedCols.includes(c) && !sweep.comparisons[0]?.inputColumns.includes(c) && v.every((x) => typeof x === "string") && new Set(v).size <= 4)
    .map(([c]) => c)
    .slice(0, 2);
  const knobCols = sweep.knobs.length ? sweep.knobs : [{ name: "scenario", values: sweep.labels }];
  const summary = row === -1;
  const each = row === -2;
  const cols = summary ? [...outcomes, ...changedCols.filter((c) => categorical(sweep.base?.[c])), ...changedCols.filter((c) => !categorical(sweep.base?.[c]))] : changedCols;
  // Side by side, the records some scenario changed come first; a few fit.
  const hit = (r: number) => sweep.comparisons.some((cmp) => cmp.output.some((o) => o.changedRows.includes(r)));
  const everyRow = Array.from({ length: rows }, (_, i) => i);
  const shown = each ? [...everyRow.filter(hit), ...everyRow.filter((r) => !hit(r))].slice(0, 3) : [row];
  const recordsOf = (list: number[]) => list.map((r) => recordLabel(r, sweep.key)).join(", ");
  const baseKnob = (name: string) => {
    const values = sweep.knobBase[name] ?? [];
    if (!summary && !each) return formatValue(values[row], knobShort(name));
    return values.every((v) => same(v, values[0])) ? formatValue(values[0], knobShort(name)) : values.map((x) => formatValue(x, knobShort(name))).join(" / ");
  };
  // Counts for an outcome ("approve 12 · decline 25"), the average for a number, each against the original run.
  const overall = (values: unknown[] | undefined, base?: unknown[], name?: string) => {
    const v = values ?? [];
    if (categorical(v)) {
      const count = (xs: unknown[] | undefined, k: string) => (xs ?? []).filter((x) => x === k).length;
      return [...new Set([...v, ...(base ?? [])] as string[])]
        .sort()
        .map((k) => {
          const d = base ? count(v, k) - count(base, k) : 0;
          return `${k} ${count(v, k)}${d ? ` (${d > 0 ? "+" : ""}${d})` : ""}`;
        })
        .join(" · ");
    }
    const nums = v.filter((x): x is number => typeof x === "number");
    if (!nums.length) return "varies";
    const avg = (xs: number[]) => xs.reduce((t, x) => t + x, 0) / xs.length;
    const baseNums = (base ?? []).filter((x): x is number => typeof x === "number");
    const d = baseNums.length ? avg(nums) - avg(baseNums) : 0;
    return `avg ${formatValue(avg(nums), name)}${Math.abs(d) > 1e-12 ? ` (${d > 0 ? "+" : "−"}${formatValue(Math.abs(d), name)})` : ""}`;
  };
  // A number's change is averaged over the records it moved, so a 3-record effect isn't diluted by 37 that didn't move.
  const avgDelta = (c: string, i: number, changedRows: number[]) => {
    const ds = changedRows.map((r) => (sweep.outputs[i]?.[c]?.[r] as number) - (sweep.base?.[c]?.[r] as number)).filter((d) => !Number.isNaN(d));
    return ds.length ? ds.reduce((t, x) => t + x, 0) / ds.length : 0;
  };
  const [metric, setMetric] = useState<string>();
  const delta = (c: string, i: number, changedRows: number[]) => {
    const d = avgDelta(c, i, changedRows);
    if (!d) return "";
    const size = isRateName(c) && Math.abs(d) < 1 ? `${Number((Math.abs(d) * 100).toFixed(2))} pp` : formatValue(Math.abs(d), c);
    return `${d > 0 ? "▲ +" : "▼ −"}${size}`;
  };
  // A scenario whose knobs equal the original run's: it is the setting in force now.
  const isCurrent = (i: number) => knobCols.every((k) => k.name === "scenario" || (sweep.knobBase[k.name] ?? []).every((v) => same(v, k.values[i])));
  const summaryLine = (i: number, c: string) => {
    const diff = sweep.comparisons[i].output.find((o) => o.name === c);
    const numeric = !categorical(sweep.base?.[c]);
    const cls = !diff ? "unchanged" : numeric ? (avgDelta(c, i, diff.changedRows) < 0 ? "changed down" : "changed up") : "changed";
    return (
      <div key={c} className={cls} title={diff ? `changed for ${recordsOf(diff.changedRows)}` : "same as the original run for every record"}>
        <span className="muted">{c}</span>{" "}
        <span className="mono">
          {!diff ? (outcomes.includes(c) ? overall(sweep.outputs[i]?.[c]) : "no change") : numeric ? `${delta(c, i, diff.changedRows)} (${diff.changedRows.length} rec.)` : overall(sweep.outputs[i]?.[c], sweep.base?.[c])}
        </span>
      </div>
    );
  };
  const summaryCell = (i: number, c: string) => {
    const diff = sweep.comparisons[i].output.find((o) => o.name === c);
    if (!diff) return <td key={c} className="unchanged mono" title="same as the original run for every record">{outcomes.includes(c) ? overall(sweep.outputs[i]?.[c]) : "no change"}</td>;
    const numeric = !categorical(sweep.base?.[c]);
    return (
      <td
        key={c}
        className={`changed mono ${numeric ? (avgDelta(c, i, diff.changedRows) < 0 ? "down" : "up") : ""}`}
        title={`changed for ${recordsOf(diff.changedRows)}; original: ${overall(sweep.base?.[c])}`}
      >
        {numeric ? (
          `${delta(c, i, diff.changedRows)} (${diff.changedRows.length})`
        ) : (
          overall(sweep.outputs[i]?.[c], sweep.base?.[c])
        )}
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
  // With exactly two knobs, a grid: one knob down, the other across, one result in each cell.
  const grid = summary && knobCols.length === 2 && knobCols.every((k) => k.name !== "scenario");
  // Offers first: they are what a sweep of rates and caps is usually about.
  const numericCols = cols.filter((c) => !categorical(sweep.base?.[c]));
  const shownMetric = metric && cols.includes(metric) ? metric : numericCols.find((c) => /offer/.test(c)) ?? numericCols[0] ?? cols[0];
  const uniq = (vs: unknown[]) => vs.filter((v, i) => vs.findIndex((w) => same(v, w)) === i);
  const cellText = (i: number, c: string) => {
    const diff = sweep.comparisons[i].output.find((o) => o.name === c);
    if (!diff) return "no change";
    return categorical(sweep.base?.[c]) ? overall(sweep.outputs[i]?.[c], sweep.base?.[c]) : `${delta(c, i, diff.changedRows)} (${diff.changedRows.length} rec.)`;
  };
  const matrix = grid && (
    <div className="sweep-scroll">
      <div className="summary">
        <span>Showing</span>
        <select aria-label="grid metric" value={shownMetric} onChange={(e) => setMetric(e.target.value)}>
          {cols.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
        <span className="muted small">original run: {overall(sweep.base?.[shownMetric], undefined, shownMetric)}</span>
      </div>
      <table className="sweep grid">
        <thead>
          <tr>
            <th>{knobShort(knobCols[0].name)} ↓ · {knobShort(knobCols[1].name)} →</th>
            {uniq(knobCols[1].values).map((v, j) => (
              <th key={j} className="mono">{formatValue(v, knobShort(knobCols[1].name))}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {uniq(knobCols[0].values).map((a, r) => (
            <tr key={r}>
              <th className="mono">{formatValue(a, knobShort(knobCols[0].name))}</th>
              {uniq(knobCols[1].values).map((b, j) => {
                const i = sweep.labels.findIndex((_, k) => same(knobCols[0].values[k], a) && same(knobCols[1].values[k], b));
                return i < 0 ? (
                  <td key={j} />
                ) : (
                  <td key={j} className={`mono clickable ${open === i ? "open" : ""} ${cellText(i, shownMetric) === "no change" ? "unchanged" : "changed"}`} title="See what changed, step by step" onClick={() => onOpen(i)}>
                    {cellText(i, shownMetric)}
                    {isCurrent(i) && <div className="current-tag">current</div>}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {outcomes.map((c) => (
        <div key={c} className="muted small">
          <span className="mono">{c}</span> is the same in every scenario: {overall(sweep.base?.[c])}.
        </div>
      ))}
    </div>
  );
  return (
    <section>
      <div className="summary">
        <span>Show</span>
        <select aria-label="scenario record" value={row} onChange={(e) => onRow(Number(e.target.value))}>
          <option value={-2}>{rows > 3 ? "the 3 records that changed most, side by side" : "each record side by side"}</option>
          <option value={-1}>a summary of all {rows} records</option>
          {Array.from({ length: rows }, (_, i) => (
            <option key={i} value={i}>the values for {recordLabel(i, sweep.key)}</option>
          ))}
        </select>
      </div>
      <p className="hint">
        {grid ? "One cell per combination; click one to see what changed, step by step." : "One row per scenario; click one to see what changed, step by step."}{summary ? " “−R 5,303.95 (3 rec.)” means 3 records changed, by −R 5,303.95 on average." : ""}
      </p>
      {cols.length === 0 ? (
        <div className="muted">No scenario changes any result.</div>
      ) : grid ? (
        matrix
      ) : (
        <div className="sweep-scroll">
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
                <th key={k.name} className="knob-col" title={k.name}>{knobShort(k.name)}</th>
              ))}
              {summary ? <th>results</th> : shown.flatMap((r) => cols.map((c) => <th key={`${c}-${r}`}>{c}</th>))}
            </tr>
          </thead>
          <tbody>
            <tr className="original">
              <td className="knob-col" colSpan={knobCols.length} title={knobCols.map((k) => `${k.name} = ${baseKnob(k.name)}`).join("\n")}>
                original run
                <div className="muted small">{knobCols.filter((k) => k.name !== "scenario").map((k) => baseKnob(k.name)).join(" · ")}</div>
              </td>
              {summary ? (
                <td className="stack">
                  {cols.map((c) => (
                    <div key={c}><span className="muted">{c}</span> <span className="mono">{overall(sweep.base?.[c], undefined, c)}</span></div>
                  ))}
                </td>
              ) : (
                shown.flatMap((r) => cols.map((c) => recordCell(null, c, r)))
              )}
            </tr>
            {sweep.labels.map((label, i) => (
              <tr key={i} className={`clickable ${open === i ? "open" : ""}`} title={`${label}: see what changed, step by step`} onClick={() => onOpen(i)}>
                {knobCols.map((k) => (
                  <td key={k.name} className="mono knob-col">{formatValue(k.values[i], knobShort(k.name))}</td>
                ))}
                {summary ? (
                  <td className="stack">
                    {isCurrent(i) && <div className="current-tag">the current setting</div>}
                    {cols.map((c) => summaryLine(i, c))}
                  </td>
                ) : (
                  shown.flatMap((r) => cols.map((c) => recordCell(i, c, r)))
                )}
                {sweep.errors[i] && <td className="error small">{sweep.errors[i]}</td>}
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}

    </section>
  );
}
