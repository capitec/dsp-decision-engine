import { useEffect, useMemo, useRef, useState } from "react";
import type { Comparison } from "../src/compare";
import {
  callNodes,
  formatValue,
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
import { FindStep } from "./FindStep";
import { Graph } from "./Graph";
import { fold } from "./layout";
import { NodePanel } from "./NodePanel";
import { Params } from "./Params";
import { Scenarios } from "./Scenarios";
import { StateTable } from "./StateTable";

declare function acquireVsCodeApi(): { postMessage(m: FromWebview): void };
const vscode = acquireVsCodeApi();
const send = (m: FromWebview) => vscode.postMessage(m);

const IDLE: RunStatus = { current: null, finished: false, finishedPaths: [], visits: {}, record: null };
type TreePath = { path: string; row: number; visited: string[]; result?: unknown[] };
// Flows up to this many steps draw fully open; bigger ones start with their groups folded.
const OPEN_ALL = 80;

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
  const [opened, setOpened] = useState<Set<string>>(new Set());
  // Said while a run is starting or re-running, until the next status arrives.
  const [pending, setPending] = useState<string>();
  // What the last skip or swap did, said in the pause banner until the run moves on.
  const [note, setNote] = useState<string>();
  const noteNext = useRef<string | undefined>(undefined);
  // The value picked with "explain a value": its breakdown stays open whichever step is selected.
  const [explained, setExplained] = useState<string>();
  const [codeDiff, setCodeDiff] = useState<string[]>([]);
  const [editsOpen, setEditsOpen] = useState(false);
  // Steps whose code was swapped mid-run: path -> the formula now running.
  const [formulas, setFormulas] = useState<Record<string, string | null>>({});
  const [graphHeight, setGraphHeight] = useState<number | null>(() => {
    try {
      return Number(localStorage.getItem("decider.graphHeight")) || null;
    } catch {
      return null;
    }
  });
  const dragSplit = (e: React.MouseEvent) => {
    const top = (e.currentTarget.previousElementSibling as HTMLElement | null)?.getBoundingClientRect().top ?? 0;
    const move = (ev: MouseEvent) => setGraphHeight(Math.max(120, ev.clientY - top));
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      setGraphHeight((h) => {
        try {
          if (h) localStorage.setItem("decider.graphHeight", String(h));
        } catch {
          // storage may be unavailable; the split still works for this view
        }
        return h;
      });
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };
  const shown = useRef<string | undefined>(undefined);

  useEffect(() => {
    const onMessage = (e: MessageEvent<ToWebview>) => {
      const m = e.data;
      switch (m.type) {
        case "describe":
          setDescribe(m.describe);
          // Starting a run describes the flow again: keep what the user was looking at.
          if (shown.current !== m.describe.pipeline) {
            setFormulas({});
            setSelected(undefined);
            setColumns(null);
            setRun(IDLE);
            setOpened(new Set());
          }
          shown.current = m.describe.pipeline;
          break;
        case "status":
          setRun(m);
          setPending(undefined);
          setNote(noteNext.current);
          if (noteNext.current === undefined) setCodeDiff([]);
          noteNext.current = undefined;
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
        case "edited":
          setCodeDiff(m.diff);
          setFormulas((f) => {
            const next = { ...f };
            if (m.restored) delete next[m.path];
            else next[m.path] = m.formula;
            return next;
          });
          break;
        case "select":
          setSelected(m.path);
          setTab("graph");
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

  const nodes = useMemo(
    () => (describe ? callNodes(describe.ir) : []).map((n) => (n.path in formulas ? { ...n, formula: formulas[n.path], formulaBefore: n.formula } : n)),
    [describe, formulas],
  );
  const selectedNode = nodes.find((n) => n.path === selected);

  // A lineage card about a column the newly selected step doesn't touch is stale; close it.
  useEffect(() => {
    if (column && column !== explained && selectedNode && !(selectedNode.inputs ?? []).includes(column) && !(selectedNode.outputs ?? []).includes(column)) setColumn(undefined);
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
  // Groups drawn open: the ones the user opened, and every one around the selection, the pause and the changes.
  const graphIr = useMemo(() => {
    if (!describe || nodes.length <= OPEN_ALL) return describe?.ir;
    const keep = [selected, run.current?.path, ...(diff ? [...diff].filter(([, s]) => s === "changed" || s === "added").map(([p]) => p) : [])].filter(Boolean) as string[];
    return fold(describe.ir, (p) => opened.has(p) || keep.some((k) => k.startsWith(`${p}/`)));
  }, [describe, nodes, opened, selected, run.current?.path, diff]);
  const toggle = (path: string) => {
    const next = new Set(opened);
    if (next.has(path) || [selected, run.current?.path].some((k) => k?.startsWith(`${path}/`))) {
      for (const p of next) if (p === path || p.startsWith(`${path}/`)) next.delete(p);
      if (selected?.startsWith(`${path}/`)) setSelected(undefined);
    } else next.add(path);
    setOpened(next);
  };
  const changedSteps = compare.comparison ? compare.comparison.steps.filter((s) => s.status === "changed" || s.status === "added").map((s) => s.path) : [];
  const changedAt = changedSteps.indexOf(selected ?? "");
  const goChanged = (dir: 1 | -1) => setSelected(changedSteps[(changedAt + dir + changedSteps.length) % changedSteps.length]);

  if (!describe) return <div className="empty">Open a pipeline file and choose “Visualise flow”.</div>;

  const edits = Object.entries(run.edits ?? {});
  const recordPicker = (
    <select aria-label="record" value={run.record ?? ""} onChange={(e) => send({ type: "record", row: e.target.value === "" ? null : Number(e.target.value) })}>
      <option value="">all {rows} records</option>
      {Array.from({ length: rows }, (_, i) => (
        <option key={i} value={i}>{recordLabel(i, keyCol)}</option>
      ))}
    </select>
  );
  const position = (path: string) => positionIn(nodes, path);
  const editLabel = ([p, a]: [string, string]) => `${p.split("/").pop()} ${a === "delete" ? "skipped" : "edited"}`;
  const compareEdits = (only?: string) =>
    send({ type: "compareEdits", label: only ? editLabel(edits.find(([p]) => p === only)!) : edits.map(editLabel).join(", "), edits: run.edits!, path: only });
  const tabButton = (t: Tab, label: string) => (
    <button className={tab === t ? "active" : ""} onClick={() => setTab(t)}>{label}</button>
  );
  const select = (path: string) => {
    setSelected(path);
    setTab("graph");
  };
  const pausedAt = run.current && !run.finished ? `${run.current.when} ${run.current.path || "the start"}` : null;
  const shownTreePath = treePath && run.record === treePath.row ? treePath : null;
  const withDetails = details && tab === "graph" && !!selectedNode;

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
        {columns && !pausedAt && <label title="Show values for one record instead of the whole batch">Focus record {recordPicker}</label>}
      </header>
      {pending && <div className="pause-banner pending">⏳ {pending}</div>}
      {!pending && pausedAt && run.current && (
        <div className="pause-banner" title={run.current.path}>
          ⏸ Paused {run.current.when} <strong>{run.current.path.split("/").pop() || "the start"}</strong>
          {columns && <> · focus {recordPicker}</>}
          {run.record !== null && columns && (
            <>
              {" "}
              <select
                aria-label="explain"
                title="How was a value computed for this record? Pick one to see its breakdown on the step that wrote it"
                value={explained ?? ""}
                onChange={(e) => {
                  setExplained(e.target.value || undefined);
                  setColumn(e.target.value || undefined);
                  setTab("graph");
                }}
              >
                <option value="">explain a value…</option>
                {columns
                  .filter((c) => c.producer !== "input")
                  .map((c) => (
                    <option key={c.name} value={c.name}>{c.name} = {formatValue(c.value, c.name)}</option>
                  ))}
              </select>
            </>
          )}
          {edits.length > 0 && (
            <>
              {" · "}
              {edits.length <= 3 ? (
                edits.map(([p, a]) => (
                  <span key={p} className="edit-chip">
                    ✎ {editLabel([p, a])}
                    {edits.length > 1 && (
                      <button className="link" title="Compare the flow as started with only this edit" onClick={() => compareEdits(p)}>compare</button>
                    )}
                  </span>
                ))
              ) : (
                <button className="link" aria-expanded={editsOpen} title="The steps skipped or swapped in this run; compare each on its own" onClick={() => setEditsOpen(!editsOpen)}>
                  ✎ {edits.length} edits {editsOpen ? "▴" : "▾"}
                </button>
              )}{" "}
              <button className="banner-button" title="Run the flow as started and as edited, start to end, and compare every result" onClick={() => compareEdits()}>
                {edits.length > 1 ? "Compare all with start" : "Compare with start"}
              </button>
            </>
          )}
          {editsOpen && edits.length > 0 && (
            <div className="edit-menu-pop">
              {edits.map(([p, a]) => (
                <div key={p}>
                  ✎ {editLabel([p, a])}{" "}
                  <button
                    className="link"
                    title="Compare the flow as started with only this edit"
                    onClick={() => {
                      setEditsOpen(false);
                      compareEdits(p);
                    }}
                  >
                    compare with start
                  </button>
                </div>
              ))}
            </div>
          )}
          {note && <div className="banner-note">{note}</div>}

        </div>
      )}
      {tab === "graph" && (
        <div className="subbar">
          <FindStep nodes={nodes} onPick={setSelected} selected={selected} />
          {nodes.length > OPEN_ALL && opened.size > 0 && (
            <button className="link" title="Fold every group back into one box" onClick={() => setOpened(new Set())}>fold all</button>
          )}
          <details className="legend-pop">
            <summary>View</summary>
            <div className="legend-body">
              <label><input type="checkbox" checked={showData} onChange={(e) => setShowData(e.target.checked)} /> show all data links</label>
              <label><input type="checkbox" checked={details} onChange={(e) => setDetails(e.target.checked)} /> step details</label>
              <span><span className="line solid" /> runs next</span>
              <span><span className="line dotted" /> passes data (for the selected step)</span>
              <span><span className="swatch paused" /> paused here</span>
              <span><span className="swatch lineage" /> inputs of the picked value</span>
              <span>✓ has run</span>
              <span>◇ decision tree</span>
              <span>▦ lookup table</span>
              <span><span className="swatch changed" /> changed in the last comparison</span>
              <span><span className="swatch added" /> added in the last comparison</span>
              <span>⊞ data frame step</span>
            </div>
          </details>
          {compare.comparison && (
            <span className="legend">
              <label title="Colour the graph by the last comparison: orange changed, green added"><input type="checkbox" checked={showDiff} onChange={(e) => setShowDiff(e.target.checked)} /> changes</label>
              {showDiff && changedSteps.length > 0 && (
                <>
                  <button title="Previous changed step" onClick={() => goChanged(-1)}>◀</button>
                  <span title={changedSteps[changedAt]}>{changedAt >= 0 ? `${changedAt + 1} of ${changedSteps.length}: ${changedSteps[changedAt].split("/").pop()}` : `${changedSteps.length} changed`}</span>
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
      {tab === "graph" && selected && (
        <div className="crumbs" title={selected}>
          {position(selected) && <span className="muted">{position(selected)} · </span>}
          {selected.split("/").map((part, i, all) => (
            <span key={i}>
              {i > 0 && <span className="muted"> › </span>}
              {i === all.length - 1 ? <strong>{part}</strong> : part}
            </span>
          ))}
        </div>
      )}
      {/* An answer that reads as text (a value's breakdown, a table's matched row) gets more of the panel than the graph. */}
      <main className={withDetails && selectedNode ? ((lineage && column) || (selectedNode.table && run.record !== null) ? "detailed explaining" : "detailed") : ""}>
        {tab === "graph" && (
          <Graph
            ir={graphIr!}
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
            onToggle={toggle}
            height={graphHeight}
          />
        )}
        {tab === "graph" && withDetails && selectedNode && <div className="splitter" title="Drag to give the graph or the details more room" onMouseDown={dragSplit} />}
        {tab === "state" && <StateTable columns={columns} record={run.record} keyCol={keyCol} selected={selectedNode} order={columnOrder} onPick={setColumn} picked={column} />}
        {tab === "params" && (
          <Params
            schema={describe.params}
            values={describe.values ?? {}}
            tables={Object.fromEntries(nodes.filter((n) => n.table).flatMap((n) => Object.keys(n.params).map((k) => [k, n.table!])))}
            inputColumns={inputColumns}
            record={run.record}
            keyCol={keyCol}
            sessionRunning={columns !== null}
            onWhatIf={(params, overrides, row, label) => send({ type: "whatIf", params, overrides, row, label })}
            onRestart={(params) => send({ type: "restartWith", params })}
            onSelectStep={select}
          />
        )}
        {tab === "scenarios" && (
          <Scenarios
            schema={describe.params}
            values={describe.values ?? {}}
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
            onFocus={columns ? (row) => send({ type: "record", row }) : undefined}
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
            onPick={(name) => {
              setExplained(undefined);
              setColumn(name);
            }}
            onSelect={setSelected}
            onReveal={(path) => send({ type: "reveal", path })}
            onRewind={(path) => send({ type: "rewind", path })}
            onRunTo={(path) => {
              setPending(`Running the flow to ${path.split("/").pop()}…`);
              send({ type: "runTo", path });
            }}
            onStep={() => send({ type: "step" })}
            comparison={showDiff ? compare.comparison : null}
            onOpenDiff={(path) => send({ type: "openDiff", path })}
            values={describe.values ?? {}}
            onSkip={(path) => {
              setPending(`Skipping ${path.split("/").pop()} and re-running from there…`);
              noteNext.current = `Skipped ${path.split("/").pop()}; re-ran from there, keeping everything before it.`;
              send({ type: "skip", path });
            }}
            onRestore={(path) => {
              setPending(`Putting the original ${path.split("/").pop()} back and re-running from there…`);
              noteNext.current = `Put the original ${path.split("/").pop()} back; continue to run it.`;
              send({ type: "restore", path });
            }}
            onReload={(path) => {
              setPending(`Reloading ${path.split("/").pop()} and re-running from there…`);
              noteNext.current = `Loaded your edited ${path.split("/").pop()} and went back to just before it; continue to run the new code.`;
              send({ type: "reloadStep", path });
            }}
          />
        )}
      </main>
    </div>
  );
}

/** "step 7 of 89 in policy" for a path in a flow of `nodes`. */
function positionIn(nodes: { path: string }[], path: string): string | null {
  const group = path.slice(0, path.lastIndexOf("/"));
  const siblings = nodes.filter((n) => n.path.slice(0, n.path.lastIndexOf("/")) === group);
  const i = siblings.findIndex((n) => n.path === path);
  return i < 0 || siblings.length < 2 ? null : `step ${i + 1} of ${siblings.length} in ${group.split("/").pop()}`;
}

function producers(l: Lineage): string[] {
  return [...(l.producer ? [l.producer] : []), ...l.inputs.flatMap(producers)];
}
