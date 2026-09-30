import { formatValue, kindLabel, lastSegment, type CallNodeJson, type IRNodeJson } from "./model/protocol";
import { findNode, positionIn, touchPoints } from "./layout";

/** What the graph, tree and inspector all agree is selected. */
export type Selection =
  | { kind: "node"; path: string }
  | { kind: "value"; name: string }
  | { kind: "edge"; id: string; from: string; to: string; label?: string; columns?: string[] };

// A value read or written by this many steps lists them all; past that, it shows a bound and a hint.
const TOUCH_BOUND = 50;

export function Inspector({ ir, selected, onSelect, onReveal, nodes }: {
  ir: IRNodeJson;
  selected?: Selection;
  onSelect: (s: Selection) => void;
  onReveal?: (path: string) => void;
  nodes: CallNodeJson[];
}) {
  if (!selected) return <EmptyHint />;
  return (
    <aside className="inspector">
      {selected.kind === "node" && <NodeBody ir={ir} path={selected.path} nodes={nodes} onSelect={onSelect} onReveal={onReveal} />}
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

function NodeBody({ ir, path, nodes, onSelect, onReveal }: {
  ir: IRNodeJson; path: string; nodes: CallNodeJson[];
  onSelect: (s: Selection) => void; onReveal?: (path: string) => void;
}) {
  const node = findNode(ir, path);
  if (!node) return null;
  const call = node.kind === "call" ? node : null;
  const group = node.kind === "call" ? null : node;
  const pos = positionIn(nodes, path);
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
      <Section title="Source">
        <div className="source-line mono">{node.file ?? "built-in"}{node.line ? `:${node.line}` : ""}</div>
        {onReveal && node.file && <button className="link" onClick={() => onReveal(node.path)}>Open source</button>}
      </Section>
    </>
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
