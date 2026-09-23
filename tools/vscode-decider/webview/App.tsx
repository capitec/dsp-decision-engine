import { useEffect, useMemo, useState } from "react";
import type { Comparison } from "../src/compare";
import {
  callNodes,
  recordLabel,
  type ColumnHistory,
  type ColumnSummary,
  type DescribeResult,
  type FromWebview,
  type Lineage,
  type RecordKey,
  type RunStatus,
  type Tab,
  type ToWebview,
} from "../src/protocol";
import type { Sweep } from "../src/sweep";
import { Compare } from "./Compare";
import { Graph } from "./Graph";
import { NodePanel } from "./NodePanel";
import { Params } from "./Params";
import { Scenarios } from "./Scenarios";
import { StateTable } from "./StateTable";

declare function acquireVsCodeApi(): { postMessage(m: FromWebview): void };
const vscode = acquireVsCodeApi();
const send = (m: FromWebview) => vscode.postMessage(m);

const IDLE: RunStatus = { current: null, finished: false, finishedPaths: [], visits: {}, record: null };
type TreePath = { path: string; row: number; visited: string[]; result?: unknown[] };

export function App() {
  const [describe, setDescribe] = useState<DescribeResult>();
  const [run, setRun] = useState<RunStatus>(IDLE);
  const [columns, setColumns] = useState<ColumnSummary[] | null>(null);
  const [rows, setRows] = useState(0);
  const [keyCol, setKeyCol] = useState<RecordKey>(null);
  const [lineage, setLineage] = useState<Lineage | null>(null);
  const [history, setHistory] = useState<ColumnHistory | null>(null);
  const [treePath, setTreePath] = useState<TreePath | null>(null);
  const [compare, setCompare] = useState<{ comparison: Comparison | null; busy?: string; error?: string }>({ comparison: null });
  const [sweep, setSweep] = useState<{ sweep: Sweep | null; busy?: string; error?: string }>({ sweep: null });
  const [selected, setSelected] = useState<string>();
  const [column, setColumn] = useState<string>();
  const [showData, setShowData] = useState(false);
  const [showDiff, setShowDiff] = useState(true);
  const [details, setDetails] = useState(true);
  const [tab, setTab] = useState<Tab>("graph");
  const [zoom, setZoom] = useState<number | "auto">("auto");

  useEffect(() => {
    const onMessage = (e: MessageEvent<ToWebview>) => {
      const m = e.data;
      switch (m.type) {
        case "describe":
          setDescribe(m.describe);
          setSelected(undefined);
          setColumns(null);
          setRun(IDLE);
          break;
        case "status":
          setRun(m);
          if (m.current) setSelected(m.current.path);
          break;
        case "state":
          setColumns(m.columns);
          setRows(m.rows);
          setKeyCol(m.key);
          break;
        case "lineage":
          setLineage(m.lineage);
          setHistory(m.history);
          break;
        case "treePath":
          setTreePath(m);
          break;
        case "compare":
          setCompare(m);
          // Start where the runs first differ.
          if (m.comparison) setSelected(m.comparison.firstDivergence ?? m.comparison.steps.find((s) => s.status !== "same" && s.status !== "not run")?.path);
          break;
        case "tab":
          setTab(m.tab);
          break;
        case "sweep":
          setSweep(m);
          break;
      }
    };
    window.addEventListener("message", onMessage);
    send({ type: "ready" });
    return () => window.removeEventListener("message", onMessage);
  }, []);

  // Lineage and history depend on the run's position and the focused record, so ask again when either moves.
  useEffect(() => {
    if (column && columns?.some((c) => c.name === column)) send({ type: "lineage", name: column });
    else {
      setLineage(null);
      setHistory(null);
    }
  }, [column, columns]);

  const nodes = useMemo(() => (describe ? callNodes(describe.ir) : []), [describe]);
  const selectedNode = nodes.find((n) => n.path === selected);

  // A lineage card about a column the newly selected step doesn't touch is stale; close it.
  useEffect(() => {
    if (column && selectedNode && !(selectedNode.inputs ?? []).includes(column) && !(selectedNode.outputs ?? []).includes(column)) setColumn(undefined);
  }, [selected]);

  // A tree's path for the focused record, once the tree has run.
  useEffect(() => {
    if (selectedNode?.callKind === "row" && run.record !== null && run.finishedPaths.includes(selectedNode.path)) send({ type: "treePath", path: selectedNode.path });
  }, [selectedNode, run.record, run.finishedPaths]);

  const columnOrder = useMemo(() => {
    const seen: string[] = [];
    for (const n of nodes) for (const c of [...(n.inputs ?? []), ...(n.outputs ?? [])]) if (!seen.includes(c)) seen.push(c);
    return seen;
  }, [nodes]);
  const lineagePaths = useMemo(() => new Set(lineage ? producers(lineage) : []), [lineage]);
  const inputColumns = useMemo(() => {
    const written = new Set(nodes.flatMap((n) => n.outputs ?? []));
    return [...new Set(nodes.flatMap((n) => n.inputs ?? []))].filter((c) => !written.has(c)).sort();
  }, [nodes]);
  const diff = useMemo(
    () => (showDiff && compare.comparison ? new Map(compare.comparison.steps.map((s) => [s.path, s.status])) : undefined),
    [compare.comparison, showDiff],
  );
  const changedSteps = compare.comparison ? compare.comparison.steps.filter((s) => s.status === "changed" || s.status === "added").map((s) => s.path) : [];
  const changedAt = changedSteps.indexOf(selected ?? "");
  const goChanged = (dir: 1 | -1) => setSelected(changedSteps[(changedAt + dir + changedSteps.length) % changedSteps.length]);

  if (!describe) return <div className="empty">Open a pipeline file and choose “Visualise flow”.</div>;

  const tabButton = (t: Tab, label: string) => (
    <button className={tab === t ? "active" : ""} onClick={() => setTab(t)}>{label}</button>
  );
  const select = (path: string) => {
    setSelected(path);
    setTab("graph");
  };
  const pausedAt = run.current && !run.finished ? `${run.current.when} ${run.current.path || "the start"}` : null;
  const shownTreePath = treePath && run.record === treePath.row ? treePath : null;
  const withDetails = details && tab === "graph";

  return (
    <div className="app">
      <header>
        <strong>{describe.pipeline}</strong>
        <nav>
          {tabButton("graph", "Graph")}
          {tabButton("state", "State")}
          {tabButton("params", "What-if")}
          {tabButton("scenarios", sweep.busy ? "Scenarios…" : "Scenarios")}
          {tabButton("compare", compare.busy ? "Compare…" : "Compare")}
        </nav>
        <button className="icon" title="Maximise the flow panel (again to restore)" onClick={() => send({ type: "maximise" })}>⤢</button>
        {columns && (
          <label title="Show values for one record instead of the whole batch">
            Focus record{" "}
            <select aria-label="record" value={run.record ?? ""} onChange={(e) => send({ type: "record", row: e.target.value === "" ? null : Number(e.target.value) })}>
              <option value="">all {rows} records</option>
              {Array.from({ length: rows }, (_, i) => (
                <option key={i} value={i}>{recordLabel(i, keyCol)}</option>
              ))}
            </select>
          </label>
        )}
        {pausedAt && <span className="badge" title="Where the debug run is paused">⏸ {pausedAt}</span>}
      </header>
      {tab === "graph" && (
        <div className="subbar">
          <span className="edge-key" title="Solid arrows: the order steps run in. Dotted: which step's output another reads (shown for the selected step).">
            <span className="line solid" /> runs next <span className="line dotted" /> data
          </span>
          <label><input type="checkbox" checked={showData} onChange={(e) => setShowData(e.target.checked)} /> show every data dependency</label>
          <label><input type="checkbox" checked={details} onChange={(e) => setDetails(e.target.checked)} /> details pane</label>
          <span className="legend">
            <span className="swatch paused" /> paused here <span className="swatch lineage" /> feeds the picked value <span>✓ ran</span>
          </span>
          {compare.comparison && (
            <span className="legend">
              <label><input type="checkbox" checked={showDiff} onChange={(e) => setShowDiff(e.target.checked)} /> compared:</label>
              <span className="swatch changed" /> changed <span className="swatch added" /> added <span className="swatch same" /> same
              {showDiff && changedSteps.length > 0 && (
                <>
                  <button title="Previous changed step" onClick={() => goChanged(-1)}>◀</button>
                  <span>{changedAt >= 0 ? `${changedAt + 1} of ${changedSteps.length}: ${changedSteps[changedAt]}` : `${changedSteps.length} changed`}</span>
                  <button title="Next changed step" onClick={() => goChanged(1)}>▶</button>
                </>
              )}
            </span>
          )}
          <span className="zoom-bar">
            <button title="Zoom out" onClick={() => setZoom((z) => Math.max(0.3, (z === "auto" ? 1 : z) / 1.25))}>−</button>
            <button title="Fit the width, keeping text readable" className={zoom === "auto" ? "primary" : ""} onClick={() => setZoom("auto")}>fit</button>
            <button title="Zoom in" onClick={() => setZoom((z) => Math.min(3, (z === "auto" ? 1 : z) * 1.25))}>+</button>
          </span>
        </div>
      )}
      <main>
        {tab === "graph" && (
          <Graph
            ir={describe.ir}
            showData={showData}
            run={run}
            selected={selected}
            highlightColumn={column}
            lineage={lineagePaths}
            diff={diff}
            treePath={shownTreePath}
            zoom={zoom}
            onSelect={setSelected}
            onOpen={(path) => send({ type: "reveal", path })}
          />
        )}
        {tab === "state" && <StateTable columns={columns} record={run.record} keyCol={keyCol} selected={selectedNode} order={columnOrder} onPick={setColumn} picked={column} />}
        {tab === "params" && (
          <Params
            schema={describe.params}
            inputColumns={inputColumns}
            record={run.record}
            keyCol={keyCol}
            sessionRunning={columns !== null}
            onWhatIf={(params, overrides, row, label) => send({ type: "whatIf", params, overrides, row, label })}
            onRestart={(params) => send({ type: "restartWith", params })}
          />
        )}
        {tab === "scenarios" && (
          <Scenarios
            schema={describe.params}
            columns={columns ? columns.map((c) => c.name) : inputColumns}
            pausedAt={pausedAt}
            record={run.record}
            keyCol={keyCol}
            rows={rows}
            result={sweep}
            onRun={(scenarios, fromHere) => send({ type: "sweep", scenarios, fromHere })}
            onOpen={(i) => setCompare({ comparison: sweep.sweep!.comparisons[i] })}
            onSelectStep={select}
            onCompareRevision={() => send({ type: "compareRevision" })}
            onOpenDiff={(path) => send({ type: "openDiff", path })}
          />
        )}
        {tab === "compare" && (
          <Compare
            {...compare}
            record={run.record}
            onSelect={select}
            onCompareRevision={() => send({ type: "compareRevision" })}
            onOpenDiff={(path) => send({ type: "openDiff", path })}
          />
        )}
        {withDetails && (
          <NodePanel
            node={selectedNode}
            nodes={nodes}
            onClose={() => setDetails(false)}
            run={run}
            columns={columns}
            keyCol={keyCol}
            column={column}
            lineage={lineage}
            history={history}
            treePath={shownTreePath}
            onPick={setColumn}
            onSelect={setSelected}
            onReveal={(path) => send({ type: "reveal", path })}
            onRewind={(path) => send({ type: "rewind", path })}
            onRunTo={(path) => send({ type: "runTo", path })}
            onStep={() => send({ type: "step" })}
            comparison={showDiff ? compare.comparison : null}
          />
        )}
      </main>
    </div>
  );
}

function producers(l: Lineage): string[] {
  return [...(l.producer ? [l.producer] : []), ...l.inputs.flatMap(producers)];
}
