import { useEffect, useRef } from "react";
import { same, type Comparison } from "../src/compare";
import { clampNote, Explain } from "./Explain";
import { header } from "./TableGrid";
import { ValueTimeline } from "./Timeline";
import { formatValue, recordLabel, type CallNodeJson, type ColumnSummary, type Lineage, type RecordKey, type RunStatus, type ValueHistory } from "../src/protocol";

interface Props {
  node: CallNodeJson;
  nodes: CallNodeJson[];
  onClose: () => void;
  run: RunStatus;
  columns: ColumnSummary[] | null;
  keyCol: RecordKey;
  column?: string;
  lineage: Lineage | null;
  history: ValueHistory | null;
  treePath: { path: string; row: number; visited: string[]; result?: unknown[] } | null;
  onPick: (name?: string) => void;
  onSelect: (path: string) => void;
  onReveal: (path: string) => void;
  onRewind: (path: string) => void;
  onGoTo: (change: number | string) => void;
  onRunTo: (path: string) => void;
  onStep: () => void;
  /** Step into the Python of the step the run is paused before. */
  onDebugStep: () => void;
  /** The comparison the graph is coloured by, if any: params show both sides. */
  comparison: Comparison | null;
  onOpenDiff: (path: string) => void;
  /** The flow's PARAMS document, where lookup tables keep their rows. */
  values: Record<string, unknown>;
  onSkip: (path: string) => void;
  onReload: (path: string) => void;
  onRestore: (path: string) => void;
  /** Breakpoint controls for the selected step. */
  controls?: React.ReactNode;
  /** The branch or loop the selected step steers, shown first. */
  groupControls?: React.ReactNode;
}

/** A decision table's rows, when `node` is one whose rows are a shared param. */
function tableOf(node: CallNodeJson, values: Record<string, unknown>): { name: string; rows: Record<string, unknown>[] } | null {
  if (!node.source.includes("DecisionTable")) return null;
  const shared = (values.shared ?? {}) as Record<string, unknown>;
  const name = Object.keys(node.params).find((k) => Array.isArray(shared[k]));
  return name ? { name, rows: shared[name] as Record<string, unknown>[] } : null;
}

/** The details pane: the selected step, then the picked column's lineage and history. */
export function NodePanel({ node, nodes, onClose, run, columns, keyCol, column, lineage, history, treePath, onPick, onSelect, onReveal, onRewind, onGoTo, onRunTo, onStep, onDebugStep, comparison, onOpenDiff, values, onSkip, onReload, onRestore, controls, groupControls }: Props) {
  const visits = run.visits[node.path];
  const who = run.record === null ? null : recordLabel(run.record, keyCol);
  const valueOf = (name: string) => {
    if (run.record === null) return undefined;
    const c = columns?.find((x) => x.name === name);
    return c ? formatValue(c.value, name) : undefined;
  };
  const path = treePath && treePath.path === node.path && run.record === treePath.row ? treePath : null;
  const paused = run.current && !run.finished;
  const ran = run.finishedPaths.includes(node.path);
  const card = lineage && lineage.name === column && !lineage.unset ? lineage : null;
  const latest = history && history.name === column ? history.changes.filter((c) => !c.pending && !c.kept).at(-1) : undefined;
  const behind = !!card && run.record !== null && latest?.value !== undefined && !same(latest.value, card.value);
  const table = tableOf(node, values);
  // A cap or floor step, once run for the focused record: whether its limit bound.
  const raw = (name: string) => columns?.find((x) => x.name === name)?.value;
  const limit =
    node.formula && run.record !== null && ran
      ? clampNote(
          node.formula,
          (node.inputs ?? []).map((i) => ({ name: i, value: raw(i) })),
          raw((node.outputs ?? [])[0]),
          Object.fromEntries([...Object.entries(node.params).map(([k, v]) => [k, (values.shared as Record<string, unknown> | undefined)?.[k] ?? v]), ...(node.inputs ?? []).map((i) => [i, raw(i)])]),
        )
      : null;
  const pane = useRef<HTMLElement>(null);
  // A newly selected step starts at the top of the pane, not where the last one was scrolled to.
  useEffect(() => {
    pane.current?.scrollTo(0, 0);
    // A lookup table's answer is its matched row: bring it into view once the pane has laid out.
    // So is an open breakdown of a value.
    const t = setTimeout(() => (pane.current?.querySelector(".table-match") ?? pane.current?.querySelector(".timeline") ?? pane.current?.querySelector(".how"))?.scrollIntoView({ block: "start" }), 60);
    return () => clearTimeout(t);
  }, [node.path, path?.row, card?.name]);
  const atThis = run.current?.path === node.path && run.current.when === "before" && !run.finished;
  const name = node.path.split("/").pop();
  const change = comparison ? comparison.steps.find((s) => s.path === node.path && s.status !== "same" && s.status !== "not run") : undefined;
  return (
    <aside ref={pane}>
      <button className="close link" title="Hide details" onClick={onClose}>✕</button>
      <div className="step-head">
        <h3>{name} {node.path !== name && <span className="muted">in {node.path.slice(0, -(name?.length ?? 0) - 1)}</span>}</h3>
        <div className="actions">
          <button onClick={() => onReveal(node.path)}>Open source</button>
          {run.edits?.[node.path] === "delete" ? null : atThis ? (
            <>
              <button className="primary" onClick={onStep} title="Run this step and pause just after it">Run through {name}</button>
              {node.python?.bodyLine && (
                <button title="Attach the Python debugger and stop on the first line of this step's code, for the focused record; step through it with F10 and F11" onClick={onDebugStep}>
                  Step into the Python
                </button>
              )}
            </>
          ) : ran && paused ? null : (
            <button className="primary" onClick={() => onRunTo(node.path)} title="Run the flow and pause just before this step">Run to {name}</button>
          )}
          {atThis && !run.edits?.[node.path] && (
            <>
              <button title="Take this step out of the paused run and re-run from where it was; the source is not changed" onClick={() => onSkip(node.path)}>Skip this step</button>
              <button title="Save your change to this step's code first: reloads the file and runs the edited step in its place, from here" onClick={() => onReload(node.path)}>Use edited code</button>
            </>
          )}
          {paused && !atThis && !run.edits?.[node.path] && (
            <details className="edit-menu">
              <summary>Change the run ▾</summary>
              <div className="edit-menu-body">
                {ran && <button onClick={() => onRewind(node.path)} title="Run the flow again from this step, keeping the values before it">Re-run from {name}</button>}
                <button title="Take this step out of the paused run and re-run from where it was; the source is not changed" onClick={() => onSkip(node.path)}>Skip {name}</button>
                <button title="Save your change to this step's code first: reloads the file and runs the edited step in its place, from here" onClick={() => onReload(node.path)}>Use edited code</button>
              </div>
            </details>
          )}
        </div>
      </div>
      {groupControls}
      {controls}
      {history && history.name === column && <ValueTimeline history={history} who={who} current={run.current} onSelect={onSelect} onGoTo={onGoTo} />}
      {card && !(path && table) && (history?.name === card.name ? (
        // Mid-loop, the breakdown follows the loop's carried value, one round behind what the history says now.
        <details className="breakdown" open={!behind || undefined}>
          <summary>How {card.name} is calculated, step by step{behind ? " (as of the loop's last finished round)" : ""}</summary>
          <Explain entry={card} who={who} nodes={nodes} values={values} onSelect={onSelect} />
        </details>
      ) : (
        <Explain entry={card} who={who} nodes={nodes} values={values} onSelect={onSelect} />
      ))}
      {who && ran && (node.outputs ?? []).length > 0 && (
        <div className="wrote">
          For {who}: {(node.outputs ?? []).map((o) => `${o} = ${valueOf(o) ?? "?"}`).join(", ")}
          {limit && <div className="clamp">{limit}</div>}
        </div>
      )}
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
      {node.formula && (node.formulaBefore !== undefined || !node.body) && (
        <div className="step-formula mono" title="What the step returns">
          returns {node.formula}
          {node.formulaBefore !== undefined && (
            <>
              <span className="edit-mark"> edited</span>
              <div className="before">was: <s>{node.formulaBefore}</s></div>
            </>
          )}
        </div>
      )}
      <div className="muted" title={node.source}>{node.table ? "lookup table, matched once per record" : KIND[node.callKind]}{node.doc ? ` · ${node.doc}` : ""}</div>
      {Object.keys(node.params).length > 0 && !table && (
        <div className="muted small mono">{Object.entries(node.params).map(([k, v]) => `${k} = ${formatValue(v, k)}`).join(" · ")}</div>
      )}
      {node.body && node.formulaBefore === undefined && (
        <details className="step-code" open={!card && node.body.split("\n").length <= 8}>
          <summary>code</summary>
          <pre>{node.body}</pre>
        </details>
      )}
      {run.edits?.[node.path] && (
        <div className="edited-note">
          {run.edits[node.path] === "delete" ? (
            <>
              Skipped in this run: the steps after it ran without it.{" "}
              {paused && <button onClick={() => onRestore(node.path)} title="Put the step back and re-run its flow from the start of that flow">Restore step</button>}
            </>
          ) : (
            <>
              Running your edited code in this run.{" "}
              {paused && <button onClick={() => onRestore(node.path)} title="Swap the code the run started with back in, and re-run from here">Use original code</button>}
            </>
          )}
        </div>
      )}
      {path && table && (
        <>
          <h4>Row for {who}</h4>
          <TableMatch table={table} expression={node.table} visited={path.visited} result={path.result} outputs={node.outputs ?? []} inputs={(node.inputs ?? []).map((i) => `${i} = ${valueOf(i) ?? "?"}`)} />
        </>
      )}
      {path && !table && (
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
                {" → "}gives <strong>{(node.outputs ?? []).map((o, i) => `${o} = ${formatValue(path.result![i], o)}`).join(", ")}</strong>
              </span>
            )}
            <div className="muted small">with {(node.inputs ?? []).map((i) => `${i} = ${valueOf(i) ?? "?"}`).join(", ")}</div>
          </div>
        </>
      )}
      <h4>Reads{who && <span className="muted"> · values for {who} · click one to see where it came from</span>}</h4>
      <Chips names={node.inputs} picked={column} onPick={onPick} valueOf={valueOf} />
      <h4>
        Writes
        {who && !ran && <span className="muted"> · not run yet</span>}
        {who && ran && atThis && run.current?.iteration ? <span className="muted"> · not run yet in iteration {run.current.iteration}; iteration {run.current.iteration - 1} wrote</span> : null}
      </h4>
      <Chips names={node.outputs} picked={column} onPick={onPick} valueOf={ran ? valueOf : () => undefined} />
      {Object.keys(node.params).length > 0 && (
        <>
          <h4>Params</h4>
          {comparison && <div className="muted">before → after</div>}
          {Object.entries(node.params).map(([k, v]) => {
            if (table?.name === k) return <div key={k} className="mono">{k} = lookup table, {table.rows.length} rows <span className="muted small">(edit its rows in What-if)</span></div>;
            if (Array.isArray(v)) return <div key={k} className="mono">{k} = table, {v.length} row{v.length === 1 ? "" : "s"} by default <span className="muted small">(edit its rows in What-if)</span></div>;
            const [a, b] = comparison ? sides(comparison, node, k, v) : [v, v];
            return (
              <div key={k} className="mono">
                {k} = {formatValue(a, k)}
                {!same(a, b) && <strong> → {formatValue(b, k)}</strong>}
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
      {card && (
        <details>
          <summary>Full lineage of {card.name}</summary>
          <LineageTree entry={card} record={run.record} onSelect={onSelect} />
        </details>
      )}
    </aside>
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

/** Which row of a lookup table a record matched, with the rows around it. */
function TableMatch({ table, expression, visited, result, outputs, inputs }: { table: { name: string; rows: Record<string, unknown>[] }; expression?: Record<string, unknown> | null; visited: string[]; result?: unknown[]; outputs: string[]; inputs: string[] }) {
  const tried = visited.map(Number).filter((n) => !Number.isNaN(n));
  const last = tried.length ? tried[tried.length - 1] : -1;
  const row = table.rows[last];
  // The reference stops at the first row that matches; if the result isn't that row's, no row matched.
  const matched = row !== undefined && outputs.every((o, i) => result === undefined || same(row[o], result[i]));
  const cols = Object.keys(table.rows[0] ?? {});
  return (
    <div className="table-match">
      <div>
        {matched ? <>Matched row <strong>{last + 1}</strong> of <span className="mono">{table.name}</span></> : <>No row of <span className="mono">{table.name}</span> matched: the default applies</>}
        {result && <> → <strong>{outputs.map((o, i) => `${o} = ${formatValue(result[i], o)}`).join(", ")}</strong></>}
      </div>
      <div className="muted small">with {inputs.join(", ")}</div>
      {matched && (
        <div className="matched-card">
          Row {last + 1}: {cols.map((c) => `${header(c, expression)} ${formatValue(row[c] ?? null, c)}`).join(" · ")}
        </div>
      )}
      <table className="table-grid">
        <thead>
          <tr>
            <th>#</th>
            {cols.map((c) => <th key={c}>{header(c, expression)}</th>)}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((r, i) => (
            <tr key={i} className={matched && i === last ? "matched" : tried.includes(i) ? "tried" : ""}>
              <td className="muted small">{i + 1}</td>
              {cols.map((c) => <td key={c} className="mono">{formatValue(r[c] ?? null, c)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
