import { useEffect, useState } from "react";
import { formatValue, lastSegment, recordLabel, walk, type Checkpoint, type Controls, type Force, type Hit, type IRNodeJson, type RecordKey, type Watch } from "../src/protocol";
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
  const how = f.arm !== undefined ? `→ ${g?.arms[f.arm] ?? `arm ${f.arm}`}` : `runs exactly ${f.iterations}×`;
  return `${lastSegment(f.path)} ${how}${forWho(f.row, keyCol)}`;
}

export function watchText(w: Watch, keyCol: RecordKey): string {
  if (w.iteration !== undefined) return `before iteration ${w.iteration} of ${lastSegment(w.path!)}`;
  const where = w.scope?.length ? ` only in ${w.scope.map((s) => s.split("/").slice(-2).join(" / ")).join(", ")}` : "";
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
  /** Where the debug run is paused. */
  current?: Checkpoint | null;
  onChange: (c: Controls) => void;
  onCompare: (a: Side, b: Side) => void;
  /** Go back to before the condition, and run on to `back` (where the run is paused) when given. */
  onRerun: (path: string, back?: Checkpoint | null) => void;
}

/** Force a branch's arm or a loop's iteration count, compare two of them, or pause at an iteration. */
export function GroupControls({ group, controls, record, keyCol, paused, ran, current, onChange, onCompare, onRerun }: GroupProps) {
  const isLoop = group.kind === "loop";
  // With a record focused, forcing and comparing start with just that record.
  const [one, setOne] = useState(record !== null);
  const [oneCompare, setOneCompare] = useState(record !== null);
  useEffect(() => {
    setOne(record !== null);
    setOneCompare(record !== null);
  }, [record]);
  const [a, setA] = useState(isLoop ? "5" : "0");
  const [b, setB] = useState(isLoop ? "10" : "1");
  const [times, setTimes] = useState("");
  const [at, setAt] = useState("");
  const row = one && record !== null ? record : null;
  // Who a force or comparison applies to: every record, or the focused one.
  const whom = (what: "force" | "compare") =>
    record === null ? (
      <span>every record</span>
    ) : (
      <select
        aria-label={`${name} ${what} for`}
        value={(what === "force" ? one : oneCompare) ? "one" : "all"}
        onChange={(e) => (what === "force" ? setOne : setOneCompare)(e.target.value === "one")}
      >
        <option value="all">every record</option>
        <option value="one">{recordLabel(record, keyCol)}</option>
      </select>
    );
  const name = lastSegment(group.path);
  const mine = controls.forces.filter((f) => f.path === group.path);
  const forced = mine.find((f) => (f.row ?? null) === row);
  const whoForced = mine[0]?.row == null ? "every record" : recordLabel(mine[0].row, keyCol);
  // A re-run with the current forces happened: they are applied until they change.
  const [rerunDone, setRerunDone] = useState(false);
  const setForce = (f?: { arm: number } | { iterations: number }) => {
    setRerunDone(false);
    const rest = controls.forces.filter((x) => !(x.path === group.path && (x.row ?? null) === row));
    onChange({ ...controls, forces: f ? [...rest, { path: group.path, row, ...f }] : rest });
  };
  const how = (v: string) => (v === "" ? null : isLoop ? { iterations: Number(v) } : { arm: Number(v) });
  const side = (v: string): Side => {
    const f = how(v);
    const label = f === null ? `${name} as it runs` : isLoop ? `${name} run ${v}×` : `${group.arms[Number(v)]} (at ${name})`;
    const who = oneCompare && record !== null ? record : null;
    return { label: (who === null ? "" : `${recordLabel(who, keyCol)}: `) + label, forces: f ? [{ path: group.path, row: who, ...f }] : [] };
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
      <h5>Force it in the debug run</h5>
      <div className="control-row">
        {isLoop ? (
          <>
            Make {whom("force")} go round exactly{" "}
            <input aria-label={`iterations for ${name}`} className="narrow" type="number" min={0} max={group.max} placeholder="e.g. 5" value={times || (forced?.iterations ?? "")} onChange={(e) => setTimes(e.target.value)} /> times{" "}
            <button disabled={times === ""} onClick={() => (setForce({ iterations: Number(times) }), setTimes(""))}>Force</button>
          </>
        ) : (
          <>
            Send {whom("force")}{" "}
            <select aria-label={`force ${name}`} value={forced?.arm ?? ""} onChange={(e) => setForce(e.target.value === "" ? undefined : { arm: Number(e.target.value) })}>
              <option value="">the way the condition says</option>
              {group.arms.map((x, i) => (
                <option key={i} value={i}>down {x}</option>
              ))}
            </select>
          </>
        )}
        {forced && <button onClick={() => setForce()}>Stop forcing</button>}
        {paused && mine.length > 0 && !rerunDone && !(current?.path === group.cond && current.when === "before") && (
          <button
            title="Go back to just before it, keeping everything earlier, run it with the force, and come back to where you are"
            onClick={() => {
              setRerunDone(true);
              onRerun(group.cond, current);
            }}
          >
            ↺ Re-run from {name} with this force
          </button>
        )}
      </div>
      {mine.length > 0 && (
        <div className="small added">
          {current?.path === group.cond && current.when === "before"
            ? `Paused just before ${lastSegment(group.cond)}: when it runs, ${whoForced} ${isLoop ? `goes round exactly ${mine[0].iterations} times` : `goes down ${group.arms[mine[0].arm ?? 0]}`}. Step or continue to see it.`
            : rerunDone && ran
              ? `Applied: ${whoForced} ${isLoop ? `went round exactly ${mine[0].iterations} times` : `went down ${group.arms[mine[0].arm ?? 0]}`} in the re-run.`
              : ran && paused
              ? `Force is set, but ${name} already ran in this pause. Re-run it now, or it applies on the next run.`
              : `Force is set: it applies the next time ${lastSegment(group.cond)} runs.`}
        </div>
      )}
      <h5>Compare two ways</h5>
      <div className="control-row">
        For {whom("compare")}: {pick(a, setA, `${name} what-if a`)} vs {pick(b, setB, `${name} what-if b`)}
        {isLoop && " iterations"}{" "}
        <button disabled={a === b} title="Run the flow both ways, start to end, and compare every result" onClick={() => onCompare(side(a), side(b))}>Compare</button>
      </div>
      <div className="muted small">
        Runs the whole flow twice from the start for every record, {oneCompare && record !== null ? `forcing only ${recordLabel(record, keyCol)}; the others run as they are` : "forcing them all"}. Your debug run is left as it is.
        {isLoop && ` A forced count overrides ${lastSegment(group.cond)}: the loop goes round exactly that many times, up to ${group.max}.`}
      </div>
      {isLoop && <h5>Break</h5>}
      {isLoop && (
        <div className="control-row">
          Pause before iteration <input aria-label={`pause ${name} at iteration`} className="narrow" type="number" min={1} max={group.max} placeholder="e.g. 3" value={at} onChange={(e) => setAt(e.target.value)} />{" "}
          <button
            disabled={!at}
            onClick={() => {
              onChange({ ...controls, watches: [...controls.watches, { path: group.path, iteration: Number(at) }] });
              setAt("");
            }}
          >
            Add breakpoint
          </button>
        </div>
      )}
      {isLoop &&
        controls.watches.map((w, i) =>
          w.path === group.path && w.iteration !== undefined ? (
            <div key={i} className="small added">
              ⏸ Pausing before iteration {w.iteration}{" "}
              <button className="link" onClick={() => onChange({ ...controls, watches: controls.watches.filter((_, j) => j !== i) })}>remove</button>
            </div>
          ) : null,
        )}
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
  const [open, setOpen] = useState(false);
  // The breakpoints this step can trigger: on a value it writes, with a scope that covers it.
  const here = controls.watches
    .map((w, i) => ({ w, i }))
    .filter(({ w }) => w.name && writes.includes(w.name) && (!w.scope?.length || w.scope.some((s) => step === s || step.startsWith(`${s}/`))));
  const add = () => {
    const w: Watch = { name, op, value: parseValue(value, "number"), scope: scope ? [scope] : undefined, row: one && record !== null ? record : null };
    onChange({ ...controls, watches: [...controls.watches, w] });
    setAdded(`Added: the run pauses when ${watchText(w, keyCol)}. It's listed at the top; × removes it.`);
    setValue("");
    setOpen(false);
  };
  return (
    <>
    <details className="watch-form" open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>⏸ Break when a value…{here.length ? ` (${here.length} active that this step can trigger)` : ""}</summary>
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
      <div className="muted small">Pauses just after a step writes {name || "it"}, the first time a record meets the condition there.</div>
    </details>
    {added && <div className="added small">{added}</div>}
    </>
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
