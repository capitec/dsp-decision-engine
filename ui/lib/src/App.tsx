import { useEffect, useMemo, useState } from "react";
import { breakpointNodes, type BreakpointKind } from "./model/breakpoints";
import { callNodes, datasetLabel, setFields, type ColumnSummary, type Controls, type DescribeResult, type FromUI, type RecordKey, type RunStatus, type ToUI } from "./model/protocol";
import { groupPaths } from "./layout";
import { Graph } from "./Graph";
import { Inspector, type Runtime, type Selection } from "./Inspector";
import { groupsOf } from "./Controls";
import { Structure } from "./Structure";
import { Experiments } from "./Experiments";
import { ExperimentSummary } from "./ExperimentSummary";
import type { ExperimentDef, ExperimentResult, ScenarioResult } from "./model/experiment";

// Flows up to this many steps start fully open; bigger ones start folded, a group at a time.
const OPEN_ALL = 80;

const IDLE: RunStatus = { current: null, finished: false, finishedPaths: [], visits: {}, record: null };

/** Messages only an editor can act on: opening source or driving a debug run. Controls that send any other are hidden. */
export type EditorMessage = "reveal" | "run" | "runTo" | "step" | "debugStep" | "rewind";

export interface AppProps {
  send: (m: FromUI) => void;
  listen: (on: (m: ToUI) => void) => () => void;
  can: ReadonlySet<EditorMessage>;
  /** Hide the in-panel structure tree — set to false when the host already shows one of its own (e.g. a native sidebar view). Default true. */
  showStructure?: boolean;
  /** Hide the inspector's breakpoint form — set to false when the host sets breakpoints elsewhere (e.g. a sidebar with its own breakpoint dots). Default true. */
  showBreakpointForm?: boolean;
  /** Called with the selected node's path whenever the graph's selection changes, so a host-side view (e.g. a sidebar tree) can mirror it. */
  onSelectionChange?: (path: string | undefined) => void;
  /** Right-clicking a step in the graph: the host shows its own context menu (view source, edit breakpoint, …), anchored at `rect`. */
  onNodeContextMenu?: (path: string, rect: DOMRect) => void;
}

/**
 * The whole flow view. The host renders it into a sized element and wires messages both ways:
 *
 *     createRoot(el).render(<App send={post} listen={subscribe} can={new Set(["reveal", "run", "step"])} />);
 */
export function App(props: AppProps) {
  return <div className="decikit"><View {...props} /></div>;
}

function View({ send, listen, can, showStructure = true, showBreakpointForm = true, onSelectionChange, onNodeContextMenu }: AppProps) {
  const [describe, setDescribe] = useState<DescribeResult>();
  const [selected, setSelected] = useState<Selection>();
  const [opened, setOpened] = useState<Set<string>>(new Set());
  const [run, setRun] = useState<RunStatus>(IDLE);
  const [columns, setColumns] = useState<ColumnSummary[] | null>(null);
  const [rows, setRows] = useState(0);
  const [keyCol, setKeyCol] = useState<RecordKey>(null);
  const [controls, setControls] = useState<Controls>({ forces: [], watches: [] });
  const [experiment, setExperiment] = useState<ExperimentResult | null>(null);
  const [summary, setSummary] = useState<{ definition: ExperimentDef; hasResult: boolean; problems: string[] } | null>(null);

  useEffect(() => {
    const stop = listen((m) => {
      switch (m.type) {
        case "describe":
          setFields(m.describe.fields);
          setDescribe(m.describe);
          setSelected(undefined);
          setRun(IDLE);
          setColumns(null);
          setOpened(callNodes(m.describe.ir).length <= OPEN_ALL ? groupPaths(m.describe.ir) : new Set());
          break;
        case "status":
          setRun(m);
          // A breakpoint that fired for some records: show the first of them.
          if (m.hit?.rows?.length && m.record === null) send({ type: "record", row: m.hit.rows[0] });
          break;
        case "state":
          setColumns(m.columns);
          setRows(m.rows);
          setKeyCol(m.key);
          break;
        case "select":
          // From the editor's Structure tree: select the node and open the groups around it.
          setSelected({ kind: "node", path: m.path });
          setOpened((o) => {
            const next = new Set(o);
            for (const p of ancestors(m.path)) next.add(p);
            return next;
          });
          break;
        case "experiment":
          setExperiment(m.experiment);
          setSummary(null);
          break;
        case "experimentSummary":
          setSummary({ definition: m.definition, hasResult: m.hasResult, problems: m.problems });
          setExperiment(null);
          break;
        case "controls":
          // Pushed from outside (e.g. a host-side breakpoint view) — mirror it, don't echo it back.
          setControls(m.controls);
          break;
      }
    });
    send({ type: "ready" });
    return stop;
  }, []);

  useEffect(() => {
    onSelectionChange?.(selected?.kind === "node" ? selected.path : undefined);
  }, [selected, onSelectionChange]);

  const nodes = useMemo(() => (describe ? callNodes(describe.ir) : []), [describe]);
  const breakpoints = useMemo(() => (describe ? breakpointNodes(describe.ir, controls.watches, run.hit ?? null) : new Map<string, BreakpointKind>()), [describe, controls.watches, run.hit]);
  const groups = useMemo(() => (describe ? groupsOf(describe.ir) : []), [describe]);
  const reveal = can.has("reveal") ? (path: string) => send({ type: "reveal", path }) : undefined;
  const changeControls = (c: Controls) => {
    setControls(c);
    send({ type: "setControls", controls: c });
  };
  const toggleBreakpoint = (path: string) => {
    const i = controls.watches.findIndex((w) => w.path === path && w.name === undefined && w.iteration === undefined);
    changeControls({ ...controls, watches: i < 0 ? [...controls.watches, { path }] : controls.watches.filter((_, j) => j !== i) });
  };

  if (!describe) return <div className="empty">Loading the flow…</div>;

  const selectedPath = selected?.kind === "node" ? selected.path : undefined;
  const open = (path: string) => opened.has(path);
  const selectNode = (path: string) => {
    setSelected({ kind: "node", path });
    setOpened((o) => {
      const next = new Set(o);
      for (const p of ancestors(path)) next.add(p);
      return next;
    });
  };
  const onSelect = (s: Selection | undefined) => {
    if (s?.kind === "node") selectNode(s.path);
    else setSelected(s);
  };
  const toggle = (path: string) =>
    setOpened((o) => {
      const next = new Set(o);
      if (next.has(path)) {
        next.delete(path);
        if (selectedPath?.startsWith(`${path}/`)) setSelected(undefined);
      } else next.add(path);
      return next;
    });

  const runtime: Runtime = { run, columns, rows, keyCol, controls, groups };

  if (summary) {
    return (
      <ExperimentSummary
        definition={summary.definition}
        hasResult={summary.hasResult}
        problems={summary.problems}
        onRun={() => send({ type: "experimentRun" })}
        onRunOnDataset={() => send({ type: "experimentRun", pickDataset: true })}
        onOpenReport={() => send({ type: "experimentOpenReport" })}
        onClose={() => setSummary(null)}
      />
    );
  }

  if (experiment) {
    return (
      <div className="app">
        <header>
          <strong className="pipeline">Experiments</strong>
          <span className="size-badge">project-owned, reproducible runs</span>
          {experiment.manifest?.input && (
            <span className="size-badge" title={experiment.manifest.input.fingerprint ? `fingerprint ${experiment.manifest.input.fingerprint}` : undefined}>
              {datasetLabel(experiment.manifest.input.dataset, experiment.manifest.input.row_count)}
            </span>
          )}
          <button className="link" style={{ marginLeft: "auto" }} onClick={() => setExperiment(null)}>back to flow</button>
        </header>
        <div className="body experiments-body">
          <Experiments
            result={experiment}
            onDebug={(scenario: ScenarioResult) =>
              send({ type: "debugScenario", params: scenario.params ?? {}, overrides: scenario.overrides ?? {}, row: scenario.row ?? null, label: scenario.name, forces: [] })
            }
          />
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header>
        <strong className="pipeline">{describe.pipeline}</strong>
        <span className="size-badge" title="The assembled flow: every node, and the steps among them">
          {describe.size.calls} steps · {describe.size.nodes} nodes
        </span>
        {run.dataset && (
          <span className="size-badge" title={run.dataset.fingerprint ? `fingerprint ${run.dataset.fingerprint}` : undefined}>
            {datasetLabel(run.dataset.name, run.dataset.rowCount)}
          </span>
        )}
      </header>
      <DebugBar run={run} can={can} send={send} />
      <div className="body">
        {showStructure && <Structure ir={describe.ir} selected={selectedPath} opened={opened} onSelect={selectNode} onToggle={toggle} />}
        <Graph ir={describe.ir} open={open} selected={selected} onSelect={onSelect} onToggle={toggle} onReveal={reveal} nodes={nodes} run={run} breakpoints={breakpoints} onToggleBreakpoint={toggleBreakpoint} onNodeContextMenu={onNodeContextMenu} />
        <Inspector ir={describe.ir} selected={selected} onSelect={onSelect} onReveal={reveal} nodes={nodes} runtime={runtime} can={can} send={send} onChangeControls={changeControls} showBreakpointForm={showBreakpointForm} />
      </div>
    </div>
  );
}

/** Where the run is paused, and the stepping controls the editor provides. */
function DebugBar({ run, can, send }: { run: RunStatus; can: ReadonlySet<EditorMessage>; send: (m: FromUI) => void }) {
  if (!run.current || run.finished) return null;
  const name = run.current.path.split("/").pop() || "the start";
  return (
    <div className="debug-bar" title={run.current.path}>
      <span className="pause-icon">⏸</span>
      <span>
        Paused <strong>{run.current.when}</strong> {name}
        {run.current.iteration ? <> in iteration <strong>{run.current.iteration}</strong></> : null}
      </span>
      {run.hit && <span className="muted small" title="Why a breakpoint paused the run">· {run.hit.text}</span>}
      {can.has("rewind") && run.finishedPaths.length > 0 && (
        <button onClick={() => send({ type: "rewind", path: run.finishedPaths[run.finishedPaths.length - 1] })} title="Go back to just before the last step ran">⏮ Step back</button>
      )}
      {can.has("step") && <button onClick={() => send({ type: "step" })} title="Step to the next checkpoint">⏭ Step</button>}
      {can.has("debugStep") && <button onClick={() => send({ type: "debugStep" })} title="Attach the Python debugger and step the step's code">Debug step</button>}
      {can.has("run") && <button className="primary" onClick={() => send({ type: "run" })} title="Run the flow from the start, pausing at breakpoints">▶ Run flow</button>}
    </div>
  );
}

/** The group paths above `path`, nearest first. */
function ancestors(path: string): string[] {
  const parts = path.split("/");
  const out: string[] = [];
  for (let i = 1; i < parts.length; i++) out.push(parts.slice(0, i).join("/"));
  return out.reverse();
}
