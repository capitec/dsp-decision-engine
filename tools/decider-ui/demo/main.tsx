import { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { App, type EditorMessage } from "../src/App";
import { findNode } from "../src/layout";
import type { Controls, FromUI, ToUI } from "../src/model/protocol";
import "../src/style.css";
import { largeFlow, pausedRun, pausedRunFor, smallFlow, experimentResult } from "./mock";
import { DATASETS, type MockDataset } from "./vscode/datasets";
import { EXPERIMENTS, type MockExperiment } from "./vscode/experiments";
import { FILES, type LensAction } from "./vscode/loan-source";
import type { ParamRow } from "./vscode/ParamTableView";
import type { ParamOverrides } from "./vscode/ParametersView";
import { BreakpointEditor, ContextMenu, breakpointAt } from "./vscode/PipelineTree";
import { VSCodeShell, type EditorTab, type SourceTab } from "./vscode/Shell";
import "./theme.css";

// A mock host: delivers one describe payload, can pause the run, and prints the messages the UI
// would send the extension (opening source, stepping) instead of acting on them.
const can = new Set<EditorMessage>(["reveal", "run", "runTo", "step", "debugStep", "rewind"]);

const IDLE = { current: null, finished: false, finishedPaths: [], visits: {}, record: null };

type Popover = { path: string; top: number; left: number };

function Demo() {
  const [activeFile, setActiveFile] = useState("loan.py");
  // Every tab open in the flow panel's editor group: pushed on open, popped on close, like any
  // other editor tab stack — the flow panel is just one more entry, not a special-cased slot.
  const [tabs, setTabs] = useState<EditorTab[]>([]);
  const [activeTabId, setActiveTabId] = useState("");
  // Same idea on the left: starts with the code tab open, Parameters pushes onto the same stack.
  const [leftTabs, setLeftTabs] = useState<SourceTab[]>([{ id: "code", kind: "code" }]);
  const [activeLeftTabId, setActiveLeftTabId] = useState("code");
  // What "Run pipeline" from the Parameters tab would use instead of each param's declared default.
  const [paramOverrides, setParamOverrides] = useState<ParamOverrides>({});
  // A table param's rows, once its full-table tab has been opened — seeded from its declared
  // default the first time, then whatever's been edited there since.
  const [tableOverrides, setTableOverrides] = useState<Record<string, Record<string, ParamRow[]>>>({});
  // Set by a mocked debug launch, cleared by closing the flow tab — the Parameters tab reads
  // edits differently while this is true (live on the paused run, not "for next run").
  const [isDebugging, setIsDebugging] = useState(false);
  // The step a paused run is sitting at, so Parameters can say an edit applies from there on.
  const [pausedAt, setPausedAt] = useState<string | null>(null);
  const [toast, setToast] = useState<string>();
  const [revealLine, setRevealLine] = useState<number>();
  const [selectedPath, setSelectedPath] = useState<string>();
  // The single source of truth for breakpoints, shared by the graph (inside App) and the
  // sidebar's Structure tree (outside it) — pushed into App with a "controls" message whenever
  // this side changes it, and pulled back out of its own "setControls" messages when it doesn't.
  const [controls, setControls] = useState<Controls>({ forces: [], watches: [] });
  const [menu, setMenu] = useState<Popover | null>(null);
  const [editing, setEditing] = useState<Popover | null>(null);
  const [activeDataset, setActiveDataset] = useState<MockDataset>(DATASETS[0]);
  const describe = useMemo(() => (activeFile === "loan.py" ? smallFlow() : largeFlow()), [activeFile]);
  const onRef = useRef<(m: ToUI) => void>(() => undefined);

  // A click outside either popover closes it, like any other VS Code context popover.
  useEffect(() => {
    if (!menu && !editing) return;
    const close = (e: MouseEvent) => {
      const el = e.target as Element;
      if (!el.closest?.(".bp-menu") && !el.closest?.(".bp-editor")) {
        setMenu(null);
        setEditing(null);
      }
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [menu, editing]);

  const flashToast = (text: string) => {
    setToast(text);
    setTimeout(() => setToast(undefined), 2500);
  };

  // Scrolls to and flashes a node's source line, standing in for editorBridge.ts's reveal().
  const highlightSource = (path: string) => {
    const line = FILES[activeFile].lineForPath[path];
    if (line == null) return;
    setRevealLine(line);
    setTimeout(() => setRevealLine(undefined), 1500);
  };

  // "View source", wherever it's triggered from: a double-click or a context menu, on the graph
  // or the sidebar — they all land here, same as editorBridge.ts's reveal() does for real.
  const revealPath = (path: string) => {
    flashToast(`opening source for ${path}`);
    highlightSource(path);
  };

  // A step's context menu, shared by the graph's node and the sidebar's dot — right-clicking
  // either shows the same options, anchored at wherever it was clicked.
  const openNodeMenu = (path: string, rect: DOMRect) => setMenu({ path, top: rect.bottom + 4, left: rect.left });

  // Pushes a breakpoint/force change into App's own controls state, the same way a real host
  // would relay a change made in its own UI back into the webview.
  const pushControls = (next: Controls) => {
    setControls(next);
    onRef.current({ type: "controls", controls: next });
  };

  const toggleBreakpoint = (path: string) => {
    const existing = breakpointAt(controls.watches, path);
    pushControls({ ...controls, watches: existing ? controls.watches.filter((w) => w !== existing) : [...controls.watches, { path }] });
  };

  const saveCondition = (path: string, expression: string) => {
    const existing = breakpointAt(controls.watches, path);
    pushControls({ ...controls, watches: [...controls.watches.filter((w) => w !== existing), { condition: expression, scope: [path] }] });
    setEditing(null);
  };

  const removeBreakpoint = (path: string) => {
    const existing = breakpointAt(controls.watches, path);
    pushControls({ ...controls, watches: controls.watches.filter((w) => w !== existing) });
    setEditing(null);
    setMenu(null);
  };

  const send = (m: FromUI) => {
    switch (m.type) {
      case "reveal":
        revealPath(m.path);
        break;
      case "setControls":
        // The graph changed a breakpoint itself (e.g. its own dot was clicked): mirror it here
        // so the sidebar's dots stay in step, without pushing it straight back.
        setControls(m.controls);
        break;
      case "run":
        onRef.current({ type: "status", ...IDLE });
        break;
      case "debugScenario":
        flashToast(`debugger run for ${m.label}${m.row == null ? "" : ` (record ${m.row})`}`);
        break;
      case "setValue":
        flashToast(`${m.name} set to ${JSON.stringify(m.value)} — not reflected in this harness's canned state`);
        break;
      case "rewind":
        flashToast(`rewinding to just before ${m.path.split("/").pop()} — no real history in this harness`);
        break;
      case "reloadStep":
        flashToast(`reloading ${m.path.split("/").pop()}'s code from disk — nothing to reload in this harness`);
        break;
      case "restore":
        flashToast(`restoring ${m.path.split("/").pop()}'s original code — nothing to restore in this harness`);
        break;
      case "skip":
        flashToast(`skipping ${m.path.split("/").pop()} — not wired in this harness`);
        break;
      case "experimentRun": {
        const note = m.pickDataset ? "picking a dataset, then running" : "running";
        flashToast(`${note} — no real bridge in this harness, showing the canned result`);
        const result = experimentResult();
        const input = { dataset: activeDataset.name, fingerprint: activeDataset.fingerprint, columns: activeDataset.columns, row_count: activeDataset.rowCount };
        onRef.current({ type: "experiment", experiment: { ...result, manifest: { ...result.manifest, input } } });
        break;
      }
      case "experimentOpenReport": {
        const result = experimentResult();
        onRef.current({ type: "experiment", experiment: result });
        break;
      }
      default:
        console.log("[demo] message to host", m);
    }
  };

  const listen = (on: (m: ToUI) => void) => {
    onRef.current = on;
    const t = setTimeout(() => on({ type: "describe", describe }), 0);
    return () => clearTimeout(t);
  };

  // Pushes a tab onto the stack (or just brings it to front if it's already open) and activates
  // it, the same way opening a file that's already open focuses its existing tab instead of
  // duplicating it.
  const openTab = (tab: EditorTab) => {
    setTabs((ts) => (ts.some((t) => t.id === tab.id) ? ts : [...ts, tab]));
    setActiveTabId(tab.id);
  };

  // Pops a tab off the stack; if it was the active one, whatever's now on top takes over.
  const closeTab = (id: string) => {
    const next = tabs.filter((t) => t.id !== id);
    setTabs(next);
    if (activeTabId === id) setActiveTabId(next[next.length - 1]?.id ?? "");
    if (id === "flow") {
      setIsDebugging(false);
      setPausedAt(null);
    }
  };

  // The same two moves, for the left (source-like) tab stack.
  const openLeftTab = (tab: SourceTab) => {
    setLeftTabs((ts) => (ts.some((t) => t.id === tab.id) ? ts : [...ts, tab]));
    setActiveLeftTabId(tab.id);
  };

  const closeLeftTab = (id: string) => {
    const next = leftTabs.filter((t) => t.id !== id);
    setLeftTabs(next);
    if (activeLeftTabId === id) setActiveLeftTabId(next[next.length - 1]?.id ?? "");
  };

  const changeParam = (path: string, name: string, value: string) => {
    setParamOverrides((o) => ({ ...o, [path]: { ...o[path], [name]: value } }));
    if (isDebugging) flashToast(`${name} set to ${value || "(default)"} — applies from ${pausedAt ?? "here"} onward`);
  };

  // Opening a table param's full view seeds it from the declared default the first time only;
  // after that, whatever's open there (including edits) is the source of truth.
  const openTable = (path: string, name: string) => {
    setTableOverrides((o) => {
      if (o[path]?.[name]) return o;
      const info = describe.params?.[path]?.[name];
      const rows = Array.isArray(info?.default) ? (info!.default as ParamRow[]) : [];
      return { ...o, [path]: { ...o[path], [name]: rows } };
    });
    openTab({ id: `table:${path}.${name}`, kind: "paramTable", path, name });
  };

  const changeTable = (path: string, name: string, rows: ParamRow[]) => {
    setTableOverrides((o) => ({ ...o, [path]: { ...o[path], [name]: rows } }));
    if (isDebugging) flashToast(`${name} updated — applies from ${pausedAt ?? "here"} onward`);
  };

  // "Run pipeline" from the Parameters tab: same mocked run as the codelens, just noting how
  // many params aren't at their declared default.
  const runPipeline = () => {
    const changed = Object.values(paramOverrides).reduce((t, byName) => t + Object.values(byName).filter((v) => v !== "").length, 0);
    runFlow();
    if (changed) flashToast(`running with ${changed} parameter override${changed === 1 ? "" : "s"}`);
  };

  // Mimics an F5 debug launch: pauses at the first breakpoint hit. Only the loan.py example
  // has a scripted pause to land on.
  const runFlow = () => {
    openTab({ id: "flow", kind: "flow" });
    if (activeFile !== "loan.py") {
      flashToast("no debug mock for this pipeline in the harness");
      return;
    }
    setIsDebugging(true);
    const { status, columns } = pausedRun();
    setPausedAt(status.current?.path ?? null);
    const dataset = { name: activeDataset.name, rowCount: activeDataset.rowCount, fingerprint: activeDataset.fingerprint, columns: activeDataset.columns };
    onRef.current({ type: "status", ...status, dataset });
    onRef.current({ type: "state", columns, rows: 2, key: { name: "client_id", values: [1, 2] } });
  };

  const whatIf = () => {
    openTab({ id: "flow", kind: "flow" });
    if (activeFile !== "loan.py") {
      flashToast("no experiment mock for this pipeline in the harness");
      return;
    }
    const result = experimentResult();
    const input = { dataset: activeDataset.name, fingerprint: activeDataset.fingerprint, columns: activeDataset.columns, row_count: activeDataset.rowCount };
    onRef.current({ type: "experiment", experiment: { ...result, manifest: { ...result.manifest, input } } });
  };

  // Opens the editor for an experiment with no run yet, or its report when the mock says it has one.
  const selectExperiment = (experiment: MockExperiment) => {
    openTab({ id: "flow", kind: "flow" });
    onRef.current({ type: "experimentSummary", definition: experiment.definition, hasResult: !!experiment.hasResult, problems: [] });
  };

  // Picking a dataset in the sidebar stands in for pickData()'s quickpick/file dialog, plus
  // opens its own tab to preview — the real extension today has no equivalent of the second part.
  const selectDataset = (name: string) => {
    const d = DATASETS.find((x) => x.name === name);
    if (!d) return;
    setActiveDataset(d);
    openTab({ id: `dataset:${d.name}`, kind: "dataset", dataset: d });
  };

  // "Debug this record" in a dataset tab: jumps straight into a debug session already focused
  // on that row, the way selecting a record then launching would in the real extension.
  const debugRecord = (dataset: MockDataset, row: number) => {
    if (activeFile !== "loan.py") {
      flashToast("no debug mock for this pipeline in the harness — open loan.py first");
      return;
    }
    setActiveDataset(dataset);
    openTab({ id: "flow", kind: "flow" });
    setIsDebugging(true);
    const { status, columns, key } = pausedRunFor(dataset, row);
    setPausedAt(status.current?.path ?? null);
    const info = { name: dataset.name, rowCount: dataset.rowCount, fingerprint: dataset.fingerprint, columns: dataset.columns };
    onRef.current({ type: "status", ...status, dataset: info });
    onRef.current({ type: "state", columns, rows: columns[0]?.rows ?? 0, key });
    flashToast(`debugging record ${row} of ${dataset.name}`);
  };

  const openFile = (name: string) => {
    setActiveFile(name);
    openLeftTab({ id: "code", kind: "code" });
  };

  const openPipeline = (label: string) => {
    const file = label === "pipeline" ? "loan.py" : "largepipeline.py";
    setActiveFile(file);
    openTab({ id: "flow", kind: "flow" });
  };

  // Mirrors decider.selectStep: select the node in the flow panel and reveal its source,
  // triggered from the sidebar's Structure tree instead of the graph itself.
  const selectNode = (path: string) => {
    onRef.current({ type: "select", path });
    highlightSource(path);
  };

  const onLens = (action: LensAction, label: string) => {
    if (action === "visualise") openTab({ id: "flow", kind: "flow" });
    else if (action === "run") runFlow();
    else if (action === "whatif") whatIf();
    else if (action === "params") openLeftTab({ id: "params", kind: "params" });
    else flashToast(`not wired in this harness: ${label}`);
  };

  const editingNode = editing ? findNode(describe.ir, editing.path) : undefined;
  const menuNode = menu ? findNode(describe.ir, menu.path) : undefined;
  const menuItems = menu
    ? [
        ...(menuNode?.kind === "call" && menuNode.file
          ? [{ label: "View source", onClick: () => { revealPath(menu.path); setMenu(null); } }]
          : []),
        { label: "Edit breakpoint…", onClick: () => { setEditing(menu); setMenu(null); } },
      ]
    : [];

  return (
    <div className="demo">
      <div className="harness-shell">
        <VSCodeShell
          activeFile={activeFile}
          revealLine={revealLine}
          describe={describe}
          selectedPath={selectedPath}
          watches={controls.watches}
          activeDataset={activeDataset}
          experiments={EXPERIMENTS}
          leftTabs={leftTabs}
          activeLeftTabId={activeLeftTabId}
          onSelectLeftTab={setActiveLeftTabId}
          onCloseLeftTab={closeLeftTab}
          paramOverrides={paramOverrides}
          onChangeParam={changeParam}
          onRunPipeline={runPipeline}
          debugging={isDebugging}
          pausedAt={pausedAt}
          tableOverrides={tableOverrides}
          onOpenTable={openTable}
          onChangeTable={changeTable}
          tabs={tabs}
          activeTabId={activeTabId}
          onOpenFile={openFile}
          onOpenPipeline={openPipeline}
          onSelectDataset={selectDataset}
          onSelectTab={setActiveTabId}
          onCloseTab={closeTab}
          onDebugRecord={debugRecord}
          onSelectExperiment={selectExperiment}
          onSelectNode={selectNode}
          onToggleDot={toggleBreakpoint}
          onDotContextMenu={openNodeMenu}
          onLens={onLens}
          onInert={(label) => flashToast(`not wired in this harness: ${label}`)}
          panel={
            <App
              key={activeFile}
              send={send}
              listen={listen}
              can={can}
              showStructure={false}
              showBreakpointForm={false}
              onSelectionChange={setSelectedPath}
              onNodeContextMenu={openNodeMenu}
            />
          }
        />
      </div>
      {menu && <ContextMenu style={{ top: menu.top, left: menu.left }} items={menuItems} />}
      {editing && editingNode && (
        <BreakpointEditor
          style={{ top: editing.top, left: editing.left }}
          label={editing.path === "" ? describe.pipeline : editing.path.split("/").pop()!}
          inputs={editingNode.kind === "call" ? editingNode.inputs ?? [] : []}
          watch={breakpointAt(controls.watches, editing.path)}
          onSave={(expression) => saveCondition(editing.path, expression)}
          onRemove={() => removeBreakpoint(editing.path)}
          onCancel={() => setEditing(null)}
        />
      )}
      {toast && <div className="demo-toast">{toast}</div>}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<Demo />);
