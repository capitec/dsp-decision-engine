import { useEffect, useMemo, useState } from "react";
import { breakpointNodes, type BreakpointKind } from "./model/breakpoints";
import { callNodes, setFields, type ColumnSummary, type Controls, type DescribeResult, type Draft, type FromUI, type RecordKey, type RunStatus, type ToUI } from "./model/protocol";
import { groupPaths } from "./layout";
import { Graph } from "./Graph";
import { Inspector, type Runtime, type Selection } from "./Inspector";
import { groupsOf } from "./Controls";
import { Structure } from "./Structure";

// Flows up to this many steps start fully open; bigger ones start folded, a group at a time.
const OPEN_ALL = 80;

const IDLE: RunStatus = { current: null, finished: false, finishedPaths: [], visits: {}, record: null };

/** Messages only an editor can act on: opening source or driving a debug run. Controls that send any other are hidden. */
export type EditorMessage = "reveal" | "run" | "runTo" | "step" | "debugStep";

export interface AppProps {
  send: (m: FromUI) => void;
  listen: (on: (m: ToUI) => void) => () => void;
  can: ReadonlySet<EditorMessage>;
}

/**
 * The whole flow view. The host renders it into a sized element and wires messages both ways:
 *
 *     createRoot(el).render(<App send={post} listen={subscribe} can={new Set(["reveal", "run", "step"])} />);
 */
export function App(props: AppProps) {
  return <div className="decider"><View {...props} /></div>;
}

function View({ send, listen, can }: AppProps) {
  const [describe, setDescribe] = useState<DescribeResult>();
  const [selected, setSelected] = useState<Selection>();
  const [opened, setOpened] = useState<Set<string>>(new Set());
  const [run, setRun] = useState<RunStatus>(IDLE);
  const [columns, setColumns] = useState<ColumnSummary[] | null>(null);
  const [rows, setRows] = useState(0);
  const [keyCol, setKeyCol] = useState<RecordKey>(null);
  const [controls, setControls] = useState<Controls>({ forces: [], watches: [] });
  const [draft, setDraft] = useState<Draft | null>(null);

  useEffect(() => {
    const stop = listen((m) => {
      switch (m.type) {
        case "describe":
          setFields(m.describe.fields);
          setDescribe(m.describe);
          setSelected(undefined);
          setRun(IDLE);
          setColumns(null);
          setDraft(null);
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
        case "draft":
          setDraft(m.draft);
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
      }
    });
    send({ type: "ready" });
    return stop;
  }, []);

  const nodes = useMemo(() => (describe ? callNodes(describe.ir) : []), [describe]);
  const breakpoints = useMemo(() => (describe ? breakpointNodes(describe.ir, controls.watches, run.hit ?? null) : new Map<string, BreakpointKind>()), [describe, controls.watches, run.hit]);
  const groups = useMemo(() => (describe ? groupsOf(describe.ir) : []), [describe]);
  const reveal = can.has("reveal") ? (path: string) => send({ type: "reveal", path }) : undefined;
  const changeControls = (c: Controls) => {
    setControls(c);
    send({ type: "setControls", controls: c });
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

  const runtime: Runtime = { run, columns, rows, keyCol, controls, groups, draft };

  return (
    <div className="app">
      <header>
        <strong className="pipeline">{describe.pipeline}</strong>
        <span className="size-badge" title="The assembled flow: every node, and the steps among them">
          {describe.size.calls} steps · {describe.size.nodes} nodes
        </span>
        <Discovery describe={describe} />
      </header>
      <DebugBar run={run} can={can} send={send} />
      <div className="body">
        <Structure ir={describe.ir} selected={selectedPath} opened={opened} onSelect={selectNode} onToggle={toggle} />
        <Graph ir={describe.ir} open={open} selected={selected} onSelect={onSelect} onToggle={toggle} onReveal={reveal} nodes={nodes} run={run} breakpoints={breakpoints} />
        <Inspector ir={describe.ir} selected={selected} onSelect={onSelect} onReveal={reveal} nodes={nodes} runtime={runtime} can={can} send={send} onChangeControls={changeControls} onDraft={() => { setDraft(null); send({ type: "draft" }); }} />
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
      {can.has("step") && <button onClick={() => send({ type: "step" })} title="Step to the next checkpoint">Step</button>}
      {can.has("debugStep") && <button onClick={() => send({ type: "debugStep" })} title="Attach the Python debugger and step the step's code">Debug step</button>}
      {can.has("run") && <button className="primary" onClick={() => send({ type: "run" })} title="Run the flow from the start, pausing at breakpoints">▶ Run flow</button>}
    </div>
  );
}

/** The entry point among the file's candidates, with the ones that were not pipelines and why. */
function Discovery({ describe }: { describe: DescribeResult }) {
  const invalid = describe.pipelines.filter((c) => c.status === "invalid");
  return (
    <details className="discovery">
      <summary>Discovery</summary>
      <div className="discovery-body">
        <div className="muted">
          Entry point <strong>{describe.pipeline}</strong> · {describe.size.calls} steps assembled from {describe.size.nodes} nodes.
        </div>
        {describe.pipelines.filter((c) => c.status !== "invalid" && c.name !== describe.pipeline).map((c) => (
          <div key={c.name}>
            <strong>{c.name}</strong> <span className="muted">{c.kind === "build" || c.kind === "factory" ? "factory" : c.kind}</span>
          </div>
        ))}
        {invalid.map((c) => (
          <div key={c.name} className="candidate-invalid">
            <s>{c.name}</s> <span className="muted">{c.reason}</span>
          </div>
        ))}
        {describe.pipelines.filter((c) => c.status !== "invalid").length <= 1 && invalid.length === 0 && (
          <div className="muted">No other candidates in this file.</div>
        )}
      </div>
    </details>
  );
}

/** The group paths above `path`, nearest first. */
function ancestors(path: string): string[] {
  const parts = path.split("/");
  const out: string[] = [];
  for (let i = 1; i < parts.length; i++) out.push(parts.slice(0, i).join("/"));
  return out.reverse();
}
