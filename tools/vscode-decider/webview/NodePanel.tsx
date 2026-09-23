import { formatValue, kindLabel, recordLabel, type CallNodeJson, type ColumnHistory, type ColumnSummary, type Lineage, type RecordKey, type RunStatus } from "../src/protocol";

interface Props {
  node?: CallNodeJson;
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
}

/** The details pane: the selected step, then the picked column's lineage and history. */
export function NodePanel({ node, run, columns, keyCol, column, lineage, history, treePath, onPick, onSelect, onReveal, onRewind, onRunTo }: Props) {
  const visits = node && run.visits[node.path];
  const who = run.record === null ? null : recordLabel(run.record, keyCol);
  const valueOf = (name: string) => {
    if (run.record === null) return undefined;
    const c = columns?.find((x) => x.name === name);
    return c ? formatValue(c.value) : undefined;
  };
  const path = treePath && node && treePath.path === node.path && run.record === treePath.row ? treePath : null;
  const paused = run.current && !run.finished;
  return (
    <aside>
      {node ? (
        <>
          <h3>{node.path}</h3>
          <div className="muted">{kindLabel(node)} step · {node.source}</div>
          <div className="actions">
            <button onClick={() => onReveal(node.path)}>Open source</button>
            <button onClick={() => onRunTo(node.path)} title="Add a breakpoint here and run the flow to it">
              {paused ? "Continue to here" : "Run to here"}
            </button>
          </div>
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
                  <span className="muted">
                    {" → "}gives {(node.outputs ?? []).map((o, i) => `${o} = ${formatValue(path.result![i])}`).join(", ")}
                  </span>
                )}
              </div>
            </>
          )}
          <h4>Reads{who && <span className="muted"> · values for {who}</span>}</h4>
          <Chips names={node.inputs} picked={column} onPick={onPick} valueOf={valueOf} />
          <h4>Writes</h4>
          <Chips names={node.outputs} picked={column} onPick={onPick} valueOf={valueOf} />
          {Object.keys(node.params).length > 0 && (
            <>
              <h4>Params</h4>
              {Object.entries(node.params).map(([k, v]) => (
                <div key={k} className="mono">{k} = {formatValue(v)}</div>
              ))}
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
      {lineage && lineage.name === column && (
        <>
          <h4>Where {lineage.name} comes from{who ? ` for ${who}` : ""}</h4>
          <LineageTree entry={lineage} record={run.record} onSelect={onSelect} />
        </>
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
