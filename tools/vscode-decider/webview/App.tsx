import { useEffect, useMemo, useState } from "react";
import { callNodes, type Checkpoint, type ColumnSummary, type DescribeResult, type FromWebview, type ToWebview } from "../src/protocol";
import { Graph } from "./Graph";
import { StateTable } from "./StateTable";

declare function acquireVsCodeApi(): { postMessage(m: FromWebview): void };
const vscode = acquireVsCodeApi();

export function App() {
  const [describe, setDescribe] = useState<DescribeResult>();
  const [current, setCurrent] = useState<Checkpoint | null>(null);
  const [finished, setFinished] = useState<string[]>([]);
  const [columns, setColumns] = useState<ColumnSummary[] | null>(null);
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
      } else if (m.type === "status") {
        setCurrent(m.current);
        setFinished(m.finishedPaths);
        if (m.current) setSelected(m.current.path);
      } else if (m.type === "state") setColumns(m.columns);
    };
    window.addEventListener("message", onMessage);
    vscode.postMessage({ type: "ready" });
    return () => window.removeEventListener("message", onMessage);
  }, []);

  const selectedNode = useMemo(() => describe && callNodes(describe.ir).find((n) => n.path === selected), [describe, selected]);

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
        {current && <span className="badge">at {current.path || "<root>"}</span>}
      </header>
      <main>
        {tab === "graph" ? (
          <Graph
            ir={describe.ir}
            showData={showData}
            current={current?.path}
            finished={finished}
            selected={selected}
            highlightColumn={column}
            onSelect={setSelected}
            onOpen={(path) => vscode.postMessage({ type: "reveal", path })}
          />
        ) : (
          <StateTable columns={columns} selected={selectedNode} onPick={setColumn} picked={column} />
        )}
        <aside>
          {selectedNode ? (
            <>
              <h3>{selectedNode.path}</h3>
              <div className="muted">{selectedNode.source}</div>
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
            </>
          ) : (
            <div className="muted">Select a node to see what it reads and writes.</div>
          )}
        </aside>
      </main>
    </div>
  );
}

function Chips({ names, picked, onPick }: { names: string[]; picked?: string; onPick: (n?: string) => void }) {
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
