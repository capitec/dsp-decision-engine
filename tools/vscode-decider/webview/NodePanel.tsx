import type { CallNodeJson, ColumnHistory, Lineage, RunStatus } from "../src/protocol";
import { kindLabel } from "../src/protocol";

interface Props {
  hidden: boolean;
  node?: CallNodeJson;
  run: RunStatus;
  column?: string;
  lineage: Lineage | null;
  history: ColumnHistory | null;
  treePath: { path: string; row: number; visited: string[] } | null;
  onPick: (name?: string) => void;
  onSelect: (path: string) => void;
  onReveal: (path: string) => void;
  onRewind: (path: string) => void;
}

/** The side panel: the selected node, then the picked column's history and lineage. */
export function NodePanel({ hidden, node, run, column, lineage, history, treePath, onPick, onSelect, onReveal, onRewind }: Props) {
  const visits = node && run.visits[node.path];
  return (
    <aside className={hidden ? "collapsed" : ""}>
      {node ? (
        <>
          <h3>{node.path}</h3>
          <div className="muted">{kindLabel(node)} · {node.source}</div>
          <button onClick={() => onReveal(node.path)}>Open source</button>
          <h4>Reads</h4>
          <Chips names={node.inputs} picked={column} onPick={onPick} />
          <h4>Writes</h4>
          <Chips names={node.outputs} picked={column} onPick={onPick} />
          {Object.keys(node.params).length > 0 && (
            <>
              <h4>Params</h4>
              <pre>{JSON.stringify(node.params, null, 1)}</pre>
            </>
          )}
          {visits && Object.keys(visits).length > 0 && (
            <>
              <h4>Visited</h4>
              {Object.entries(visits).map(([loc, n]) => (
                <div key={loc} className="mono">#{loc} · {n} rows</div>
              ))}
            </>
          )}
          {treePath && treePath.path === node.path && (
            <>
              <h4>Path of record {treePath.row}</h4>
              <div className="mono">{treePath.visited.map((l) => `#${l}`).join(" → ")}</div>
            </>
          )}
        </>
      ) : (
        <div className="muted">Select a node to see what it reads and writes.</div>
      )}
      {history && history.name === column && (
        <>
          <h4>History of {column}</h4>
          <ol className="history">
            {history.versions.map((v, i) => (
              <li key={i}>
                {v.written === false ? (
                  <span className="muted">not on this record</span>
                ) : (
                  <span className="mono">{JSON.stringify(run.record === null ? v.values.slice(0, 3) : v.values[0])}</span>
                )}
                {" ← "}
                {v.producer === "input" || v.producer.startsWith("override@") ? (
                  <span className="muted">{v.producer}</span>
                ) : (
                  <>
                    <a onClick={() => onSelect(v.producer)}>{v.producer}</a>{" "}
                    <button className="link" title="Re-run from this step with the current values" onClick={() => onRewind(v.producer)}>rewind</button>
                  </>
                )}
              </li>
            ))}
          </ol>
        </>
      )}
      {lineage && lineage.name === column && (
        <>
          <h4>Lineage of {lineage.name}{run.record !== null ? `, record ${run.record}` : ""}</h4>
          <LineageTree entry={lineage} record={run.record} onSelect={onSelect} />
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
        {record !== null && <span className="mono"> = {JSON.stringify(entry.value)}</span>}
        {" ← "}
        {entry.producer ? <a onClick={() => onSelect(entry.producer!)}>{entry.producer}</a> : <span className="muted">input</span>}
        {entry.via && <span className="muted"> ({entry.via})</span>}
        {entry.inputs.map((i, k) => (
          <LineageTree key={k} entry={i} record={record} onSelect={onSelect} />
        ))}
      </li>
    </ul>
  );
}

function Chips({ names, picked, onPick }: { names: string[] | null; picked?: string; onPick: (n?: string) => void }) {
  if (names === null) return <div className="muted">unknown until it runs</div>;
  return (
    <div className="chips">
      {names.map((n) => (
        <button key={n} className={`chip ${picked === n ? "picked" : ""}`} onClick={() => onPick(picked === n ? undefined : n)}>
          {n}
        </button>
      ))}
    </div>
  );
}
