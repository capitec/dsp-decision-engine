import { useEffect, useRef } from "react";
import { same, type Comparison } from "../src/compare";
import { formatValue, recordLabel, type CallNodeJson, type ColumnHistory, type ColumnSummary, type Lineage, type RecordKey, type RunStatus } from "../src/protocol";

interface Props {
  node?: CallNodeJson;
  nodes: CallNodeJson[];
  onClose: () => void;
  run: RunStatus;
  columns: ColumnSummary[] | null;
  keyCol: RecordKey;
  column?: string;
  lineage: Lineage | null;
  history: ColumnHistory | null;
  treePath: { path: string; row: number; visited: string[]; result?: unknown[] } | null;
  onPick: (name?: string) => void;
  onSelect: (path: string) => void;
  onReveal: (path: string) => void;
  onRewind: (path: string) => void;
  onRunTo: (path: string) => void;
  onStep: () => void;
  /** The comparison the graph is coloured by, if any: params show both sides. */
  comparison: Comparison | null;
  onOpenDiff: (path: string) => void;
}

/** The details pane: the selected step, then the picked column's lineage and history. */
export function NodePanel({ node, nodes, onClose, run, columns, keyCol, column, lineage, history, treePath, onPick, onSelect, onReveal, onRewind, onRunTo, onStep, comparison, onOpenDiff }: Props) {
  const visits = node && run.visits[node.path];
  const who = run.record === null ? null : recordLabel(run.record, keyCol);
  const valueOf = (name: string) => {
    if (run.record === null) return undefined;
    const c = columns?.find((x) => x.name === name);
    return c ? formatValue(c.value) : undefined;
  };
  const path = treePath && node && treePath.path === node.path && run.record === treePath.row ? treePath : null;
  const paused = run.current && !run.finished;
  const ran = !!node && run.finishedPaths.includes(node.path);
  const card = lineage && lineage.name === column ? lineage : null;
  const atThis = !!node && run.current?.path === node.path && run.current.when === "before" && !run.finished;
  const name = node?.path.split("/").pop();
  const change = comparison && node ? comparison.steps.find((s) => s.path === node.path && s.status !== "same" && s.status !== "not run") : undefined;
  return (
    <aside>
      <button className="close link" title="Hide details" onClick={onClose}>✕</button>
      {node ? (
        <>
          <h3>{node.path}</h3>
          <div className="muted" title={node.source}>{KIND[node.callKind]}</div>
          <div className="actions">
            <button onClick={() => onReveal(node.path)}>Open source</button>
            {atThis ? (
              <button className="primary" onClick={onStep} title="Run this step and pause just after it">Run through {name}</button>
            ) : ran && paused ? (
              <button onClick={() => onRewind(node.path)} title="Run the flow again from this step, keeping the values before it">Re-run from {name}</button>
            ) : (
              <button className="primary" onClick={() => onRunTo(node.path)} title="Run the flow and pause just before this step">Run to {name}</button>
            )}
          </div>
          {change && (
            <div className="changed-box">
              <div className="how-title">Changed in this comparison</div>
              {change.paramChanges.map((p) => (
                <div key={p} className="mono">{p}</div>
              ))}
              {change.structural.includes("code") && (
                <div>
                  code changed{comparison!.files && <> · <a onClick={() => onOpenDiff(node.path)}>view diff</a></>}
                </div>
              )}
              {change.outputs.flatMap((o) =>
                o.samples.map((sm) => (
                  <div key={`${o.name}-${sm.row}`} className="mono">
                    {o.name} for {recordLabel(sm.row, keyCol ?? comparison!.key)}: {formatValue(sm.a)} → <strong>{formatValue(sm.b)}</strong>
                  </div>
                )),
              )}
            </div>
          )}
          {path && (
            <>
              <h4>Path for {who}</h4>
              <div className="tree-path">
                {path.visited.map((l, i) => (
                  <span key={i}>
                    {i > 0 && <span className="muted"> → </span>}
                    <span className="mono">{l}</span>
                  </span>
                ))}
                {path.result && (
                  <span>
                    {" → "}gives <strong>{(node.outputs ?? []).map((o, i) => `${o} = ${formatValue(path.result![i])}`).join(", ")}</strong>
                  </span>
                )}
                <div className="muted small">with {(node.inputs ?? []).map((i) => `${i} = ${valueOf(i) ?? "?"}`).join(", ")}</div>
              </div>
            </>
          )}
          <h4>Reads{who && <span className="muted"> · values for {who}</span>}</h4>
          <Chips names={node.inputs} picked={column} onPick={onPick} valueOf={valueOf} />
          {card && (node.inputs ?? []).includes(card.name) && (
            <HowComputed entry={card} who={who} role="an input to this step" nodes={nodes} onPick={onPick} onSelect={onSelect} />
          )}
          <h4>Writes{who && !ran && <span className="muted"> · not run yet</span>}</h4>
          <Chips names={node.outputs} picked={column} onPick={onPick} valueOf={ran ? valueOf : () => undefined} />
          {card && !(node.inputs ?? []).includes(card.name) && (
            <HowComputed entry={card} who={who} role={ran ? "written by this step" : "the value so far"} nodes={nodes} onPick={onPick} onSelect={onSelect} />
          )}
          {Object.keys(node.params).length > 0 && (
            <>
              <h4>Params</h4>
              {comparison && <div className="muted">baseline → variant</div>}
              {Object.entries(node.params).map(([k, v]) => {
                const [a, b] = comparison ? sides(comparison, node, k, v) : [v, v];
                return (
                  <div key={k} className="mono">
                    {k} = {formatValue(a)}
                    {!same(a, b) && <strong> → {formatValue(b)}</strong>}
                  </div>
                );
              })}
            </>
          )}
          {!path && visits && Object.keys(visits).length > 0 && (
            <>
              <h4>Tree nodes reached (all records)</h4>
              {Object.entries(visits).map(([loc, n]) => (
                <div key={loc} className="mono">{loc} · {n} record{n === 1 ? "" : "s"}</div>
              ))}
              {run.record === null && <div className="muted small">Focus a record to see its own path.</div>}
            </>
          )}
        </>
      ) : (
        <div className="muted">Click a step in the graph to see what it reads and writes.</div>
      )}
      {!node && card && <HowComputed entry={card} who={who} role="" nodes={nodes} onPick={onPick} onSelect={onSelect} />}
      {card && (
        <details>
          <summary>Full lineage of {card.name}</summary>
          <LineageTree entry={card} record={run.record} onSelect={onSelect} />
        </details>
      )}
      {history && history.name === column && (
        <>
          <h4>Every value of {column} so far</h4>
          <ol className="history">
            {history.versions.map((v, i) => (
              <li key={i}>
                {v.written === false ? (
                  <span className="muted">not on this record</span>
                ) : (
                  <span className="mono">{run.record === null ? v.values.slice(0, 3).map(formatValue).join(", ") : formatValue(v.values[0])}</span>
                )}
                {" ← "}
                {v.producer === "input" || v.producer.startsWith("override@") ? (
                  <span className="muted">{v.producer === "input" ? "input" : "set by you"}</span>
                ) : (
                  <>
                    <a onClick={() => onSelect(v.producer)}>{v.producer}</a>{" "}
                    <button className="link" title="Re-run from this step with the current values" onClick={() => onRewind(v.producer)}>re-run from here</button>
                  </>
                )}
              </li>
            ))}
          </ol>
        </>
      )}
    </aside>
  );
}

/** "term_cap = 60, written by term/term_cap from requested_term = 72 and ceiling = 60". */
function HowComputed({ entry, who, role, nodes, onPick, onSelect }: { entry: Lineage; who: string | null; role: string; nodes: CallNodeJson[]; onPick: (n?: string) => void; onSelect: (p: string) => void }) {
  const producer = nodes.find((n) => n.path === entry.producer);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => box.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }), [entry]);
  return (
    <div className="how" ref={box}>
      <div className="how-title">
        <span className="mono">{entry.name}{who ? ` = ${formatValue(entry.value)}` : ""}</span>
        {role && <span className="muted"> ({role})</span>} comes from{who ? ` for ${who}` : ""}:
      </div>
      {entry.producer === null ? (
        <div>It is an input: it arrives with the data.</div>
      ) : (
        <>
          <div>
            Written by <a onClick={() => onSelect(entry.producer!)}>{entry.producer}</a>
            {entry.via === "merge" && <span className="muted"> (the branch arm this record took)</span>}
            {entry.via === "carry" && <span className="muted"> (the loop's last iteration)</span>}
            {entry.inputs.length > 0 && " from:"}
          </div>
          <ul>
            {entry.inputs.map((i, k) => (
              <li key={k}>
                <a className="mono" title={`How is ${i.name} computed?`} onClick={() => onPick(i.name)}>{i.name}</a>
                {who && <span className="mono"> = {formatValue(i.value)}</span>}
                <span className="muted"> {i.producer ? `from ${i.producer}` : "input"}</span>
              </li>
            ))}
            {producer &&
              Object.entries(producer.params).map(([k, v]) => (
                <li key={k}>
                  <span className="mono">{k} = {formatValue(v)}</span> <span className="muted">parameter</span>
                </li>
              ))}
          </ul>
          <div className="muted small">Click an input to go one step further back.</div>
        </>
      )}
    </div>
  );
}

function LineageTree({ entry, record, onSelect }: { entry: Lineage; record: number | null; onSelect: (p: string) => void }) {
  return (
    <ul className="lineage">
      <li>
        <span className="mono">{entry.name}</span>
        {record !== null && <span className="mono"> = {formatValue(entry.value)}</span>}
        {" ← "}
        {entry.producer ? <a onClick={() => onSelect(entry.producer!)}>{entry.producer}</a> : <span className="muted">input</span>}
        {entry.via && <span className="muted"> ({entry.via === "merge" ? "branch result" : "loop result"})</span>}
        {entry.inputs.map((i, k) => (
          <LineageTree key={k} entry={i} record={record} onSelect={onSelect} />
        ))}
      </li>
    </ul>
  );
}

function Chips({ names, picked, onPick, valueOf }: { names: string[] | null; picked?: string; onPick: (n?: string) => void; valueOf: (n: string) => string | undefined }) {
  if (names === null) return <div className="muted">unknown until it runs</div>;
  return (
    <div className="chips">
      {names.map((n) => {
        const v = valueOf(n);
        return (
          <button key={n} className={`chip ${picked === n ? "picked" : ""}`} title="Show where this value comes from" onClick={() => onPick(picked === n ? undefined : n)}>
            {n}
            {v !== undefined && <span className="chip-value"> = {v}</span>}
          </button>
        );
      })}
    </div>
  );
}

const KIND: Record<CallNodeJson["callKind"], string> = {
  scalar: "Python step, run once per record",
  row: "decision tree, walked once per record",
  frame: "data frame step, run on the whole batch",
};

/** A param's value in the baseline and the variant: from the params documents, or the defaults each side declared. */
function sides(c: Comparison, node: CallNodeJson, name: string, value: unknown): [unknown, unknown] {
  const pick = (doc: unknown) => node.path.split("/").reduce<unknown>((d, part) => (d as Record<string, unknown> | undefined)?.[part], doc) as Record<string, unknown> | undefined;
  const declared = c.steps.find((s) => s.path === node.path)?.paramChanges.find((p) => p.startsWith(`${name}: `));
  const parse = (x: string) => {
    try {
      return JSON.parse(x);
    } catch {
      return x;
    }
  };
  const [da, db] = declared ? declared.slice(name.length + 2).split(" → ").map(parse) : [value, value];
  return [pick(c.paramsDocs?.a)?.[name] ?? da, pick(c.paramsDocs?.b)?.[name] ?? db];
}
