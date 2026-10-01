import { useState } from "react";
import { formatValue, kindLabel, lastSegment, previewOf, type CallNodeJson, type ColumnSummary, type Controls, type FromUI, type IRNodeJson, type RecordKey, type RunStatus } from "./model/protocol";
import { findNode, positionIn, touchPoints } from "./layout";
import { GroupControls, steered, WatchForm, type Group } from "./Controls";
import { parseValue } from "./Params";

/** What the graph, tree and inspector all agree is selected. */
export type Selection =
  | { kind: "node"; path: string }
  | { kind: "value"; name: string }
  | { kind: "edge"; id: string; from: string; to: string; label?: string; columns?: string[] };

/** The run's live state, shared by the inspector's runtime views. */
export interface Runtime {
  run: RunStatus;
  columns: ColumnSummary[] | null;
  rows: number;
  keyCol: RecordKey;
  controls: Controls;
  groups: Group[];
}

// A value read or written by this many steps lists them all; past that, it shows a bound and a hint.
const TOUCH_BOUND = 50;

export function Inspector({ ir, selected, onSelect, onReveal, nodes, runtime, can, send, onChangeControls, showBreakpointForm = true }: {
  ir: IRNodeJson;
  selected?: Selection;
  onSelect: (s: Selection) => void;
  onReveal?: (path: string) => void;
  nodes: CallNodeJson[];
  runtime: Runtime;
  can: ReadonlySet<string>;
  send: (m: FromUI) => void;
  onChangeControls: (c: Controls) => void;
  /** Hide the breakpoint form — set to false when the host sets breakpoints elsewhere. Default true. */
  showBreakpointForm?: boolean;
}) {
  if (!selected) return <EmptyHint />;
  return (
    <aside className="inspector">
      {selected.kind === "node" && (
        <NodeBody ir={ir} path={selected.path} nodes={nodes} onSelect={onSelect} onReveal={onReveal} runtime={runtime} can={can} send={send} onChangeControls={onChangeControls} showBreakpointForm={showBreakpointForm} />
      )}
      {selected.kind === "value" && <ValueBody ir={ir} name={selected.name} onSelect={onSelect} />}
      {selected.kind === "edge" && <EdgeBody selected={selected} nodes={nodes} onSelect={onSelect} />}
    </aside>
  );
}

function EmptyHint() {
  return (
    <aside className="inspector empty">
      <p>Select a step to see what it reads and writes, an edge to see the values it carries, or a value to see every step that touches it.</p>
    </aside>
  );
}

function NodeBody({ ir, path, nodes, onSelect, onReveal, runtime, can, send, onChangeControls, showBreakpointForm }: {
  ir: IRNodeJson; path: string; nodes: CallNodeJson[];
  onSelect: (s: Selection) => void; onReveal?: (path: string) => void;
  runtime: Runtime; can: ReadonlySet<string>; send: (m: FromUI) => void;
  onChangeControls: (c: Controls) => void; showBreakpointForm: boolean;
}) {
  const node = findNode(ir, path);
  if (!node) return null;
  const call = node.kind === "call" ? node : null;
  const group = node.kind === "call" ? null : node;
  const pos = positionIn(nodes, path);
  const outputNames = [...new Set(nodes.flatMap((n) => n.outputs ?? []))].sort();
  const steeredGroup = call && steered(runtime.groups, call.path);
  return (
    <>
      <div className="inspector-head">
        <h2>{lastSegment(path)}</h2>
        <span className="kind-label">{call ? kindLabel(call) : node.kind}</span>
        {pos && <span className="muted">{pos}</span>}
      </div>
      {call?.doc && <p className="doc">{call.doc}</p>}
      {call?.formula && <pre className="formula">returns {call.formula}</pre>}
      {call ? (
        <>
          <Section title="Reads" hint="click a value to find every step that touches it">
            <Chips names={call.inputs} onPick={(n) => onSelect({ kind: "value", name: n })} />
          </Section>
          <Section title="Writes">
            <Chips names={call.outputs} onPick={(n) => onSelect({ kind: "value", name: n })} />
          </Section>
          {Object.keys(call.params).length > 0 && (
            <Section title="Params">
              <div className="params-list">
                {Object.entries(call.params).map(([k, v]) => <div key={k} className="mono">{k} = {formatValue(v, k)}</div>)}
              </div>
            </Section>
          )}
          {call.body && (
            <details className="section">
              <summary>Code</summary>
              <pre className="code">{call.body}</pre>
            </details>
          )}
        </>
      ) : (
        <Section title="Contains">
          <div className="muted">{group!.folded?.length ?? group!.children.length} step{(group!.folded?.length ?? group!.children.length) === 1 ? "" : "s"}</div>
        </Section>
      )}
      {call && <RuntimeSection call={call} runtime={runtime} onSelect={onSelect} onReveal={onReveal} send={send} />}
      {call && (
        <>
          {showBreakpointForm && <BreakpointSection call={call} outputNames={outputNames} runtime={runtime} onChangeControls={onChangeControls} />}
          {steeredGroup && (
            <GroupControls
              key={steeredGroup.path}
              group={steeredGroup}
              controls={runtime.controls}
              record={runtime.run.record}
              keyCol={runtime.keyCol}
              paused={!!runtime.run.current && !runtime.run.finished}
              ran={runtime.run.finishedPaths.includes(steeredGroup.cond)}
              current={runtime.run.current}
              onChange={onChangeControls}
              onCompare={(a, b) => send({ type: "compareForces", a, b })}
              onRerun={(path, back) => send({ type: "rerun", path, back: back?.path, when: back?.when })}
            />
          )}
        </>
      )}
    </>
  );
}

/** The node's place in the run: paused here, run, or still to come — and its live values. */
function RuntimeSection({ call, runtime, onSelect, onReveal, send }: {
  call: CallNodeJson; runtime: Runtime; onSelect: (s: Selection) => void; onReveal?: (path: string) => void; send: (m: FromUI) => void;
}) {
  const { run, columns } = runtime;
  const [editing, setEditing] = useState<string>();
  const [editValue, setEditValue] = useState("");
  if (!run.current && !run.finished) return null;
  const paused = run.current?.path === call.path;
  const ran = runtime.run.finishedPaths.includes(call.path);
  const edits = runtime.run.edits?.[call.path];
  const name = (n: string) => columns?.find((c) => c.name === n);
  const shown = (n: string) => {
    const c = name(n);
    if (!c) return "—";
    const v = run.record === null ? previewOf(c, n) : formatValue(c.value, n);
    const mine = (c.producer?.startsWith("override@") || c.producer?.startsWith("force@") || c.writtenBy?.startsWith("override@") || c.changedBy?.startsWith("override@")) ? " · set by you" : "";
    return `${v}${mine}`;
  };
  const startEdit = (n: string) => {
    const c = name(n);
    setEditing(n);
    setEditValue(c ? String(run.record === null ? previewOf(c, n) : formatValue(c.value, n)) : "");
  };
  const saveEdit = (n: string) => {
    send({ type: "setValue", name: n, value: parseValue(editValue, name(n)?.dtype) });
    setEditing(undefined);
  };
  const row = (n: string, cls: string) => {
    const c = name(n);
    const editable = paused && !!c;
    return (
      <div key={n} className={cls}>
        <button className="link" onClick={() => onSelect({ kind: "value", name: n })}>{n}</button>{" "}
        {editing === n ? (
          <>
            =
            <input
              aria-label={`override ${n}`}
              className="value-edit"
              autoFocus
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") saveEdit(n);
                if (e.key === "Escape") setEditing(undefined);
              }}
            />
            <button className="link" title="Save" onClick={() => saveEdit(n)}>✓</button>
            <button className="link" title="Cancel" onClick={() => setEditing(undefined)}>✕</button>
          </>
        ) : (
          <>
            = {shown(n)}
            {editable && <button className="link edit-value" title="Override this value" onClick={() => startEdit(n)}>✎</button>}
          </>
        )}
      </div>
    );
  };
  return (
    <Section title="Runtime" hint={paused ? "the run is paused here" : ran ? "this step has run" : "not yet run"}>
      {paused && (
        <div className="paused-here">
          ⏸ Paused <strong>{run.current!.when}</strong>
          {run.current!.iteration ? <> in iteration <strong>{run.current!.iteration}</strong></> : null}
        </div>
      )}
      {edits && <div className="muted small">{edits === "delete" ? "skipped this session" : "code replaced this session"}</div>}
      {(call.inputs ?? []).map((n) => row(n, "mono value-row"))}
      {(call.outputs ?? []).map((n) => row(n, "mono value-row writes"))}
      {paused && (
        <div className="runtime-actions">
          {onReveal && <button className="link" onClick={() => onReveal(call.path)}>✎ Replace step…</button>}
          <button className="link" title="Pick up this step's code from its source file, as it's saved now" onClick={() => send({ type: "reloadStep", path: call.path })}>↺ Reload edited code</button>
          {edits && <button className="link" onClick={() => send({ type: "restore", path: call.path })}>Restore original</button>}
          <button className="link" onClick={() => send({ type: "skip", path: call.path })}>⊘ Skip this step</button>
        </div>
      )}
      {runtime.run.visits[call.path] && Object.keys(runtime.run.visits[call.path]).length > 0 && (
        <details className="trace">
          <summary>trace · {Object.keys(runtime.run.visits[call.path]).length} position{Object.keys(runtime.run.visits[call.path]).length === 1 ? "" : "s"} visited</summary>
          {Object.entries(runtime.run.visits[call.path]).map(([loc, rows]) => (
            <div key={loc} className="mono value-row">#{loc} · {rows} row{rows === 1 ? "" : "s"}</div>
          ))}
        </details>
      )}
      {!runtime.run.trace?.available && (
        <div className="muted small trace-off" title={runtime.run.trace?.reason}>{runtime.run.trace?.reason ?? "trace capture is off; showing live state"}.</div>
      )}
    </Section>
  );
}

/** Break on this step, on a value it writes, or (for a loop) before an iteration. */
function BreakpointSection({ call, outputNames, runtime, onChangeControls }: {
  call: CallNodeJson; outputNames: string[]; runtime: Runtime; onChangeControls: (c: Controls) => void;
}) {
  return (
    <WatchForm
      key={call.path}
      names={outputNames}
      step={call.path}
      writes={call.outputs ?? []}
      scopes={call.path.split("/").map((_, i, parts) => parts.slice(0, i + 1).join("/"))}
      name={call.outputs?.[0]}
      record={runtime.run.record}
      keyCol={runtime.keyCol}
      controls={runtime.controls}
      onChange={onChangeControls}
    />
  );
}

function ValueBody({ ir, name, onSelect }: { ir: IRNodeJson; name: string; onSelect: (s: Selection) => void }) {
  const touches = touchPoints(ir, name);
  const writers = touches.filter((n) => (n.outputs ?? []).includes(name));
  const readers = touches.filter((n) => (n.inputs ?? []).includes(name));
  return (
    <>
      <div className="inspector-head">
        <h2>{name}</h2>
        <span className="kind-label">value</span>
      </div>
      <Section title={`Written by ${writers.length}`}>
        <NodeList nodes={writers} onSelect={(p) => onSelect({ kind: "node", path: p })} />
      </Section>
      <Section title={`Read by ${readers.length}`}>
        <NodeList nodes={readers} onSelect={(p) => onSelect({ kind: "node", path: p })} bound />
      </Section>
    </>
  );
}

function EdgeBody({ selected, nodes, onSelect }: { selected: Selection & { kind: "edge" }; nodes: CallNodeJson[]; onSelect: (s: Selection) => void }) {
  const from = nodes.find((n) => n.path === selected.from);
  const to = nodes.find((n) => n.path === selected.to);
  return (
    <>
      <div className="inspector-head">
        <h2>{selected.label || "runs before"}</h2>
        <span className="kind-label">edge</span>
      </div>
      <Section title="Between">
        <NodeList nodes={[from, to].filter(Boolean) as CallNodeJson[]} onSelect={(p) => onSelect({ kind: "node", path: p })} />
      </Section>
      {selected.columns && selected.columns.length > 0 && (
        <Section title="Carries">
          <Chips names={selected.columns} onPick={(n) => onSelect({ kind: "value", name: n })} />
        </Section>
      )}
    </>
  );
}

function Section({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <section className="section">
      <h3>{title}{hint && <span className="muted small"> · {hint}</span>}</h3>
      {children}
    </section>
  );
}

function Chips({ names, onPick }: { names: string[] | null; onPick: (n: string) => void }) {
  if (names === null) return <div className="muted">unknown</div>;
  if (names.length === 0) return <div className="muted">nothing</div>;
  return (
    <div className="chips">
      {names.map((n) => <button key={n} className="chip" onClick={() => onPick(n)}>{n}</button>)}
    </div>
  );
}

function NodeList({ nodes, onSelect, bound }: { nodes: CallNodeJson[]; onSelect: (p: string) => void; bound?: boolean }) {
  const shown = bound ? nodes.slice(0, TOUCH_BOUND) : nodes;
  return (
    <div className="node-list">
      {shown.map((n) => (
        <button key={n.path} className="node-link" onClick={() => onSelect(n.path)} title={n.path}>
          {lastSegment(n.path)}
        </button>
      ))}
      {bound && nodes.length > TOUCH_BOUND && <div className="muted small">… and {nodes.length - TOUCH_BOUND} more. Select the value in the graph for the full fan-out.</div>}
    </div>
  );
}
