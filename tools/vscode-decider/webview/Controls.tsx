import { useState } from "react";
import { formatValue, lastSegment, recordLabel, walk, type Controls, type Force, type Hit, type IRNodeJson, type RecordKey, type Watch } from "../src/protocol";
import { parseValue } from "./Params";

export interface Group {
  path: string;
  kind: "branch" | "loop";
  /** The condition step that picks the arm or decides whether to go round again. */
  cond: string;
  /** A branch's arms, by the name of the step or flow each runs. */
  arms: string[];
  max?: number;
}

export type Side = { label: string; forces: Force[] };

export function groupsOf(ir: IRNodeJson): Group[] {
  const out: Group[] = [];
  walk(ir, (n) => {
    if (n.kind === "branch" || n.kind === "loop") out.push({ path: n.path, kind: n.kind, cond: n.children[0]?.path ?? "", arms: n.children.slice(1).map((c) => lastSegment(c.path)), max: n.maxIterations });
  });
  return out;
}

/** The branch whose condition `path` is, or the innermost loop around it: what its details can steer. */
export const steered = (groups: Group[], path: string) =>
  groups
    .filter((g) => g.cond === path || (g.kind === "loop" && path.startsWith(`${g.path}/`)))
    .sort((a, b) => b.path.length - a.path.length)[0];

const forWho = (row: number | null | undefined, keyCol: RecordKey) => (row == null ? "" : ` for ${recordLabel(row, keyCol)}`);

export function forceText(f: Force, groups: Group[], keyCol: RecordKey): string {
  const g = groups.find((x) => x.path === f.path);
  const how = f.arm !== undefined ? `down ${g?.arms[f.arm] ?? `arm ${f.arm}`}` : `exactly ${f.iterations}×`;
  return `${lastSegment(f.path)} ${how}${forWho(f.row, keyCol)}`;
}

export function watchText(w: Watch, keyCol: RecordKey): string {
  if (w.iteration !== undefined) return `before iteration ${w.iteration} of ${lastSegment(w.path!)}`;
  const where = w.scope?.length ? ` in ${w.scope.map(lastSegment).join(", ")}` : "";
  return `${w.name} ${w.op} ${formatValue(w.value, w.name)}${where}${forWho(w.row, keyCol)}`;
}

/** Why a breakpoint paused the run; `watch` (the breakpoint as set) words it the way its chip does. */
export function hitText(hit: Hit, keyCol: RecordKey, watch?: Watch): string {
  const name = watch?.name ?? hit.text.split(" ")[0];
  const who = (hit.rows ?? []).map((r, i) => `${recordLabel(r, keyCol)} = ${formatValue(hit.values?.[i], name)}`);
  const text = watch && watch.name ? `${watch.name} ${watch.op} ${formatValue(watch.value, watch.name)}` : hit.text;
  return `${text}${hit.path ? ` after ${lastSegment(hit.path)}` : ""}${who.length ? `: ${who.slice(0, 3).join(", ")}${who.length > 3 ? ` and ${who.length - 3} more` : ""}` : ""}`;
}

/** The forces and breakpoints in effect, each removable. */
export function ControlsBar({ controls, groups, keyCol, onChange }: { controls: Controls; groups: Group[]; keyCol: RecordKey; onChange: (c: Controls) => void }) {
  const { forces, watches } = controls;
  if (!forces.length && !watches.length) return null;
  return (
    <div className="controls-bar">
      {forces.map((f, i) => (
        <span key={`f${i}`} className="chip force" title="Forced: records go this way whatever the condition says">
          ⇢ {forceText(f, groups, keyCol)}{" "}
          <button className="link" aria-label={`stop forcing ${f.path}`} onClick={() => onChange({ ...controls, forces: forces.filter((_, j) => j !== i) })}>×</button>
        </span>
      ))}
      {watches.map((w, i) => (
        <span key={`w${i}`} className="chip watch" title="Breakpoint: the run pauses here">
          ⏸ {watchText(w, keyCol)}{" "}
          <button className="link" aria-label={`remove breakpoint ${watchText(w, keyCol)}`} onClick={() => onChange({ ...controls, watches: watches.filter((_, j) => j !== i) })}>×</button>
        </span>
      ))}
    </div>
  );
}

interface GroupProps {
  group: Group;
  controls: Controls;
  record: number | null;
  keyCol: RecordKey;
  /** A debug run is paused: a force can be re-run from the branch or loop. */
  paused: boolean;
  /** The condition has run in this debug run. */
  ran: boolean;
  onChange: (c: Controls) => void;
  onCompare: (a: Side, b: Side) => void;
  onRerun: (path: string) => void;
}

/** Force a branch's arm or a loop's iteration count, compare two of them, or pause at an iteration. */
export function GroupControls({ group, controls, record, keyCol, paused, ran, onChange, onCompare, onRerun }: GroupProps) {
  const isLoop = group.kind === "loop";
  const [one, setOne] = useState(false);
  const [a, setA] = useState(isLoop ? "5" : "0");
  const [b, setB] = useState(isLoop ? "10" : "1");
  const [times, setTimes] = useState("");
  const [at, setAt] = useState("");
  const [added, setAdded] = useState<string>();
  const row = one && record !== null ? record : null;
  const name = lastSegment(group.path);
  const mine = controls.forces.filter((f) => f.path === group.path);
  const current = mine.find((f) => (f.row ?? null) === row);
  const setForce = (f?: { arm: number } | { iterations: number }) => {
    const rest = controls.forces.filter((x) => !(x.path === group.path && (x.row ?? null) === row));
    onChange({ ...controls, forces: f ? [...rest, { path: group.path, row, ...f }] : rest });
  };
  const how = (v: string) => (v === "" ? null : isLoop ? { iterations: Number(v) } : { arm: Number(v) });
  const side = (v: string): Side => {
    const f = how(v);
    const label = f === null ? `${name} as it runs` : isLoop ? `${name} ${v}×` : `${name} down ${group.arms[Number(v)]}`;
    return { label: label + forWho(row, keyCol), forces: f ? [{ path: group.path, row, ...f }] : [] };
  };
  const pick = (v: string, set: (v: string) => void, label: string) =>
    isLoop ? (
      <input aria-label={label} className="narrow" type="number" min={0} max={group.max} placeholder="as it runs" value={v} onChange={(e) => set(e.target.value)} />
    ) : (
      <select aria-label={label} value={v} onChange={(e) => set(e.target.value)}>
        <option value="">as the condition says</option>
        {group.arms.map((x, i) => (
          <option key={i} value={i}>down {x}</option>
        ))}
      </select>
    );

  return (
    <section className="group-controls">
      <h4>
        {isLoop ? "Loop" : "Branch"} {name}
        {isLoop && <span className="muted small"> · up to {group.max} iterations</span>}
      </h4>
      {record !== null && (
        <label className="small">
          <input type="checkbox" checked={one} onChange={(e) => setOne(e.target.checked)} /> only {recordLabel(record, keyCol)} (forcing and comparing below)
        </label>
      )}
      <h5>Force it in the debug run</h5>
      <div className="control-row">
        {isLoop ? (
          <>
            Run it exactly{" "}
            <input aria-label={`iterations for ${name}`} className="narrow" type="number" min={0} max={group.max} value={times || (current?.iterations ?? "")} onChange={(e) => setTimes(e.target.value)} /> times{" "}
            <button disabled={times === ""} onClick={() => (setForce({ iterations: Number(times) }), setTimes(""))}>Force</button>
          </>
        ) : (
          <>
            Send {row === null ? "every record" : recordLabel(row, keyCol)}{" "}
            <select aria-label={`force ${name}`} value={current?.arm ?? ""} onChange={(e) => setForce(e.target.value === "" ? undefined : { arm: Number(e.target.value) })}>
              <option value="">the way the condition says</option>
              {group.arms.map((x, i) => (
                <option key={i} value={i}>down {x}</option>
              ))}
            </select>
          </>
        )}
        {current && <button className="link" onClick={() => setForce()}>stop forcing</button>}
        {paused && mine.length > 0 && (
          <button title="Go back to just before it, keeping everything earlier, so the force applies" onClick={() => onRerun(group.path)}>↺ Re-run {name} forced</button>
        )}
      </div>
      {mine.length > 0 && (
        <div className="small added">
          {ran && paused
            ? `On. ${name} has already run in this pause, so the force applies when it runs again: re-run it now, or on the next run.`
            : `On. It applies when the run reaches ${name}.`}
        </div>
      )}
      <h5>Compare two ways</h5>
      <div className="control-row">
        {pick(a, setA, `${name} what-if a`)} vs {pick(b, setB, `${name} what-if b`)}
        {isLoop && " iterations"}{" "}
        <button disabled={a === b} title="Run the flow both ways, start to end, and compare every result" onClick={() => onCompare(side(a), side(b))}>Compare</button>
      </div>
      <div className="muted small">Runs the whole flow twice from the start, for {row === null ? "every record" : recordLabel(row, keyCol)}; your debug run is left as it is.</div>
      {isLoop && (
        <div className="control-row">
          Pause before iteration <input aria-label={`pause ${name} at iteration`} className="narrow" type="number" min={1} max={group.max} placeholder="k" value={at} onChange={(e) => setAt(e.target.value)} />{" "}
          <button
            disabled={!at}
            onClick={() => {
              onChange({ ...controls, watches: [...controls.watches, { path: group.path, iteration: Number(at) }] });
              setAdded(`Added: the run pauses before iteration ${at}. It's listed at the top; × removes it.`);
              setAt("");
            }}
          >
            Add breakpoint
          </button>
        </div>
      )}
      {isLoop && added && <div className="added small">{added}</div>}
    </section>
  );
}

const OPS: NonNullable<Watch["op"]>[] = ["==", "!=", "<", "<=", ">", ">="];

interface WatchProps {
  names: string[];
  /** The selected step, and what it writes. */
  step: string;
  writes: string[];
  /** Where the breakpoint may be limited to: the flows holding the selected step, and the step itself. */
  scopes: string[];
  name?: string;
  record: number | null;
  keyCol: RecordKey;
  controls: Controls;
  onChange: (c: Controls) => void;
}

/** Pause the first time a record's value meets a condition, anywhere or only inside some steps. */
export function WatchForm({ names, step, writes, scopes, name: initial, record, keyCol, controls, onChange }: WatchProps) {
  const [name, setName] = useState(initial ?? names[0] ?? "");
  const [op, setOp] = useState<NonNullable<Watch["op"]>>(">=");
  const [value, setValue] = useState("");
  const [scope, setScope] = useState("");
  const [one, setOne] = useState(false);
  const [added, setAdded] = useState<string>();
  // The breakpoints this step can trigger: on a value it writes, with a scope that covers it.
  const here = controls.watches
    .map((w, i) => ({ w, i }))
    .filter(({ w }) => w.name && writes.includes(w.name) && (!w.scope?.length || w.scope.some((s) => step === s || step.startsWith(`${s}/`))));
  const add = () => {
    const w: Watch = { name, op, value: parseValue(value, "number"), scope: scope ? [scope] : undefined, row: one && record !== null ? record : null };
    onChange({ ...controls, watches: [...controls.watches, w] });
    setAdded(`Added: the run pauses when ${watchText(w, keyCol)}. It's listed at the top; × removes it.`);
    setValue("");
  };
  return (
    <details className="watch-form" open={here.length > 0 || undefined}>
      <summary>Break when a value…{here.length ? ` (${here.length} here)` : ""}</summary>
      {here.map(({ w, i }) => (
        <div key={i} className="small">
          ⏸ {watchText(w, keyCol)}{" "}
          <button className="link" onClick={() => onChange({ ...controls, watches: controls.watches.filter((_, j) => j !== i) })}>remove</button>
        </div>
      ))}
      <div className="control-row">
        <select aria-label="break when name" value={name} onChange={(e) => setName(e.target.value)}>
          {names.map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
        <select aria-label="break when op" value={op} onChange={(e) => setOp(e.target.value as typeof op)}>
          {OPS.map((o) => (
            <option key={o}>{o}</option>
          ))}
        </select>
        <input aria-label="break when value" className="narrow" placeholder="e.g. 25% or 0.25" value={value} onChange={(e) => setValue(e.target.value)} />
        {value.trim() !== "" && <span className="muted small">= {formatValue(parseValue(value, "number"), name)}</span>}
      </div>
      <div className="control-row">
        <select aria-label="break when scope" value={scope} onChange={(e) => setScope(e.target.value)}>
          <option value="">anywhere in the flow</option>
          {scopes.map((s) => (
            <option key={s} value={s}>only in {s.split("/").slice(-2).join(" / ")}</option>
          ))}
        </select>
        {record !== null && (
          <label className="small">
            <input type="checkbox" checked={one} onChange={(e) => setOne(e.target.checked)} /> only {recordLabel(record, keyCol)}
          </label>
        )}
        <button disabled={!name || value.trim() === ""} onClick={add}>Add breakpoint</button>
      </div>
      {added ? <div className="added small">{added}</div> : <div className="muted small">Pauses just after a step writes {name || "it"}, the first time a record meets the condition there.</div>}
    </details>
  );
}

/** The What-if tab's forces: per branch an arm, per loop an iteration count; empty runs as the condition says. */
export function ForcePicker({ groups, value, onChange }: { groups: Group[]; value: Record<string, string>; onChange: (v: Record<string, string>) => void }) {
  if (!groups.length) return null;
  const set = (path: string, v: string) => onChange({ ...value, [path]: v });
  return (
    <section>
      <h4>Force branches and loops</h4>
      <table>
        <tbody>
          {groups.map((g) => (
            <tr key={g.path} className={value[g.path] ? "edited" : ""}>
              <td className="name" title={g.path}>{lastSegment(g.path)} <span className="muted small">{g.kind}</span></td>
              <td>
                {g.kind === "branch" ? (
                  <select aria-label={`force ${g.path}`} value={value[g.path] ?? ""} onChange={(e) => set(g.path, e.target.value)}>
                    <option value="">as the condition says</option>
                    {g.arms.map((x, i) => (
                      <option key={i} value={i}>always down {x}</option>
                    ))}
                  </select>
                ) : (
                  <>
                    <input aria-label={`force ${g.path}`} className="narrow" type="number" min={0} max={g.max} placeholder="as it runs" value={value[g.path] ?? ""} onChange={(e) => set(g.path, e.target.value)} /> times
                  </>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export const pickedForces = (groups: Group[], value: Record<string, string>, row: number | null): Force[] =>
  groups
    .filter((g) => (value[g.path] ?? "") !== "")
    .map((g) => (g.kind === "branch" ? { path: g.path, arm: Number(value[g.path]), row } : { path: g.path, iterations: Number(value[g.path]), row }));
