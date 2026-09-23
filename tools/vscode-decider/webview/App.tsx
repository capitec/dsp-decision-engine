import { useEffect, useMemo, useState } from "react";
import {
  callNodes,
  kindLabel,
  type ColumnSummary,
  type DescribeResult,
  type FromWebview,
  type Lineage,
  type RunStatus,
  type ToWebview,
} from "../src/protocol";
import { Graph } from "./Graph";
import { StateTable } from "./StateTable";

declare function acquireVsCodeApi(): { postMessage(m: FromWebview): void };
const vscode = acquireVsCodeApi();

const IDLE: RunStatus = { current: null, finished: false, finishedPaths: [], visits: {}, record: null };

export function App() {
  const [describe, setDescribe] = useState<DescribeResult>();
  const [run, setRun] = useState<RunStatus>(IDLE);
  const [columns, setColumns] = useState<ColumnSummary[] | null>(null);
  const [rows, setRows] = useState(0);
  const [lineage, setLineage] = useState<Lineage | null>(null);
  const [selected, setSelected] = useState<string>();
  const [column, setColumn] = useState<string>();
  const [showData, setShowData] = useState(true);
  const [tab, setTab] = useState<"graph" | "state">("graph");

  useEffect(() => {
    const onMessage = (e: MessageEvent<ToWebview>) => {
      const m = e.data;
      if (m.type === "describe") {
        setDescribe(m.describe);
        setSelected(undefined);
        setColumns(null);
        setRun(IDLE);
      } else if (m.type === "status") {
        setRun(m);
        if (m.current) setSelected(m.current.path);
      } else if (m.type === "state") {
        setColumns(m.columns);
        setRows(m.rows);
      } else setLineage(m.lineage);
    };
    window.addEventListener("message", onMessage);
    vscode.postMessage({ type: "ready" });
    return () => window.removeEventListener("message", onMessage);
  }, []);

  // Lineage depends on the run's position and the focused record, so ask again whenever either moves.
  useEffect(() => {
    if (column && columns?.some((c) => c.name === column)) vscode.postMessage({ type: "lineage", name: column });
    else setLineage(null);
  }, [column, columns]);

  const selectedNode = useMemo(() => describe && callNodes(describe.ir).find((n) => n.path === selected), [describe, selected]);
  const lineagePaths = useMemo(() => new Set(lineage ? producers(lineage) : []), [lineage]);

  if (!describe) return <div className="empty">Open a pipeline file and choose “Visualise flow”.</div>;

  return (
    <div className="app">
      <header>
        <strong>{describe.pipeline}</strong>
        <nav>
          <button className={tab === "graph" ? "active" : ""} onClick={() => setTab("graph")}>Graph</button>
          <button className={tab === "state" ? "active" : ""} onClick={() => setTab("state")}>
            State{columns ? ` (${columns.length})` : ""}
          </button>
        </nav>
        {tab === "graph" && (
          <label>
            <input type="checkbox" checked={showData} onChange={(e) => setShowData(e.target.checked)} /> data edges
          </label>
        )}
        {columns && (
          <label>
            record{" "}
            <select
              aria-label="record"
              value={run.record ?? ""}
              onChange={(e) => vscode.postMessage({ type: "record", row: e.target.value === "" ? null : Number(e.target.value) })}
            >
              <option value="">all {rows}</option>
              {Array.from({ length: rows }, (_, i) => (
                <option key={i} value={i}>{i}</option>
              ))}
            </select>
          </label>
        )}
        {run.current && <span className="badge">{run.current.when} {run.current.path || "<root>"}</span>}
      </header>
      <main>
        {tab === "graph" ? (
          <Graph
            ir={describe.ir}
            showData={showData}
            run={run}
            selected={selected}
            highlightColumn={column}
            lineage={lineagePaths}
            onSelect={setSelected}
            onOpen={(path) => vscode.postMessage({ type: "reveal", path })}
          />
        ) : (
          <StateTable columns={columns} record={run.record} selected={selectedNode} onPick={setColumn} picked={column} />
        )}
        <aside>
          {selectedNode ? (
            <>
              <h3>{selectedNode.path}</h3>
              <div className="muted">{kindLabel(selectedNode)} · {selectedNode.source}</div>
              <button onClick={() => vscode.postMessage({ type: "reveal", path: selectedNode.path })}>Open source</button>
              <h4>Reads</h4>
              <Chips names={selectedNode.inputs} picked={column} onPick={setColumn} />
              <h4>Writes</h4>
              <Chips names={selectedNode.outputs} picked={column} onPick={setColumn} />
              {Object.keys(selectedNode.params).length > 0 && (
                <>
                  <h4>Params</h4>
                  <pre>{JSON.stringify(selectedNode.params, null, 1)}</pre>
                </>
              )}
              {run.visits[selectedNode.path] && Object.keys(run.visits[selectedNode.path]).length > 0 && (
                <>
                  <h4>Visited</h4>
                  {Object.entries(run.visits[selectedNode.path]).map(([loc, n]) => (
                    <div key={loc} className="mono">#{loc} · {n} rows</div>
                  ))}
                </>
              )}
            </>
          ) : (
            <div className="muted">Select a node to see what it reads and writes.</div>
          )}
          {lineage && (
            <>
              <h4>Lineage of {lineage.name}{run.record !== null ? `, record ${run.record}` : ""}</h4>
              <LineageTree entry={lineage} record={run.record} onSelect={setSelected} />
            </>
          )}
        </aside>
      </main>
    </div>
  );
}

function producers(l: Lineage): string[] {
  return [...(l.producer ? [l.producer] : []), ...l.inputs.flatMap(producers)];
}

function LineageTree({ entry, record, onSelect }: { entry: Lineage; record: number | null; onSelect: (p: string) => void }) {
  return (
    <ul className="lineage">
      <li>
        <span className="mono">{entry.name}</span>
        {record !== null && <span className="mono"> = {JSON.stringify(entry.value)}</span>}
        {" ← "}
        {entry.producer ? (
          <a onClick={() => onSelect(entry.producer!)}>{entry.producer}</a>
        ) : (
          <span className="muted">input</span>
        )}
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
