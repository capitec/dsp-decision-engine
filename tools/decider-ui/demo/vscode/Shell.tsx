import { useEffect, useState, type ReactNode } from "react";
import type { DescribeResult, Watch } from "../../src/model/protocol";
import { DatasetView } from "./DatasetView";
import { DATASETS, type MockDataset } from "./datasets";
import type { MockExperiment } from "./experiments";
import { FILES, PIPELINES, type LensAction } from "./loan-source";
import { Pane } from "./Pane";
import { ParamTableView, type ParamRow } from "./ParamTableView";
import { ParametersView, type ParamOverrides } from "./ParametersView";
import { PipelineTree } from "./PipelineTree";
import "./shell.css";

// A decorative stand-in for the VS Code window: an activity bar with the real Explorer plus a
// mock "decider" container, a file tree, a mock editor with real CodeLens buttons, and a flow
// panel that opens beside it. Nothing here is wired to an editor; it only has to move like one,
// closely enough that the flow between "pick a pipeline" / "open a file" / "run it" / "a node
// reveals its source" can be iterated on without launching the real extension.

const KEYWORDS = new Set(["def", "class", "return", "if", "else", "raise", "import", "from", "for", "in", "and", "or", "not", "True", "False", "None", "as"]);

function tokenize(line: string): { text: string; cls?: string }[] {
  const re = /(#.*$)|("""[^"]*"""|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(@[A-Za-z_]\w*)|(\b\d+\.?\d*\b)|([A-Za-z_]\w*)|(\s+)|(.)/g;
  const tokens: { text: string; cls?: string }[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(line))) {
    if (m[1]) tokens.push({ text: m[1], cls: "tok-comment" });
    else if (m[2]) tokens.push({ text: m[2], cls: "tok-string" });
    else if (m[3]) tokens.push({ text: m[3], cls: "tok-decorator" });
    else if (m[4]) tokens.push({ text: m[4], cls: "tok-number" });
    else if (m[5]) tokens.push({ text: m[5], cls: KEYWORDS.has(m[5]) ? "tok-keyword" : undefined });
    else tokens.push({ text: m[0] });
  }
  return tokens;
}

// Two tab stacks, each holding its own distinct resource types — same as a real VS Code editor
// group holds tabs for distinct resources rather than one tab that reskins itself. The left
// group is source-like views (today: the mock code view and Parameters); the right is the flow
// panel and one entry per dataset being previewed. Each tab has its own id, so several dataset
// tabs can be open side by side, independent of whether the flow tab is.
export type SourceTab = { id: "code"; kind: "code" } | { id: "params"; kind: "params" };
export type EditorTab =
  | { id: "flow"; kind: "flow" }
  | { id: string; kind: "dataset"; dataset: MockDataset }
  | { id: string; kind: "paramTable"; path: string; name: string };

export interface VSCodeShellProps {
  activeFile: string;
  /** Line to scroll to and flash in the active file, mimicking a "reveal" jump. */
  revealLine?: number;
  /** The visualised flow, once one exists — drives the sidebar's Structure tree and Parameters tab. */
  describe?: DescribeResult;
  /** The node selected in the flow panel, so the sidebar's Structure tree can highlight it too. */
  selectedPath?: string;
  /** The breakpoints in effect, shared with the flow panel's graph so the two sets of dots agree. */
  watches: Watch[];
  /** The dataset the next run would use, so the sidebar and the flow panel's chip agree. */
  activeDataset: MockDataset;
  /** Saved experiment.yaml assets and their past runs, for the sidebar's Experiments & Results pane. */
  experiments: MockExperiment[];
  /** Every tab open on the left (source-like views). */
  leftTabs: SourceTab[];
  activeLeftTabId: string;
  onSelectLeftTab: (id: string) => void;
  onCloseLeftTab: (id: string) => void;
  /** The values a "Run pipeline" from the Parameters tab would use instead of each param's default. */
  paramOverrides: ParamOverrides;
  onChangeParam: (path: string, name: string, value: string) => void;
  onRunPipeline: () => void;
  /** A debug session is paused: the Parameters tab reads edits as live, not "for next run". */
  debugging: boolean;
  /** The step a paused run is sitting at — an edit applies from there on, never retroactively. */
  pausedAt: string | null;
  /** A table param's rows, by its declaring path and name — a declared default until opened and edited. */
  tableOverrides: Record<string, Record<string, ParamRow[]>>;
  onOpenTable: (path: string, name: string) => void;
  onChangeTable: (path: string, name: string, rows: ParamRow[]) => void;
  /** Every tab open on the right (the flow panel, and dataset previews). */
  tabs: EditorTab[];
  activeTabId: string;
  onOpenFile: (name: string) => void;
  onOpenPipeline: (label: string) => void;
  onSelectDataset: (name: string) => void;
  onSelectTab: (id: string) => void;
  onCloseTab: (id: string) => void;
  onDebugRecord: (dataset: MockDataset, row: number) => void;
  onSelectExperiment: (experiment: MockExperiment) => void;
  onSelectNode: (path: string) => void;
  onToggleDot: (path: string) => void;
  onDotContextMenu: (path: string, rect: DOMRect) => void;
  onLens: (action: LensAction, label: string) => void;
  onInert: (label: string) => void;
  /** The flow tab's content — only ever one of these, unlike dataset tabs. */
  panel: ReactNode;
}

export function VSCodeShell({
  activeFile,
  revealLine,
  describe,
  selectedPath,
  watches,
  activeDataset,
  experiments,
  leftTabs,
  activeLeftTabId,
  onSelectLeftTab,
  onCloseLeftTab,
  paramOverrides,
  onChangeParam,
  onRunPipeline,
  debugging,
  pausedAt,
  tableOverrides,
  onOpenTable,
  onChangeTable,
  tabs,
  activeTabId,
  onOpenFile,
  onOpenPipeline,
  onSelectDataset,
  onSelectTab,
  onCloseTab,
  onDebugRecord,
  onSelectExperiment,
  onSelectNode,
  onToggleDot,
  onDotContextMenu,
  onLens,
  onInert,
  panel,
}: VSCodeShellProps) {
  const [container, setContainer] = useState<"explorer" | "decider">("explorer");
  const [pipelineQuery, setPipelineQuery] = useState("");
  const [experimentQuery, setExperimentQuery] = useState("");
  const file = FILES[activeFile];
  const panelOpen = tabs.length > 0;
  const showLeft = leftTabs.length > 0;

  useEffect(() => {
    if (revealLine != null) document.getElementById(`code-line-${revealLine}`)?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [revealLine]);

  const filteredPipelines = PIPELINES.filter(
    (p) => p.label.toLowerCase().includes(pipelineQuery.toLowerCase()) || p.file.toLowerCase().includes(pipelineQuery.toLowerCase()),
  );
  const filteredExperiments = experiments.filter((e) => e.name.toLowerCase().includes(experimentQuery.toLowerCase()));

  const lensByLine = new Map(file.codeLenses.map((l) => [l.line, l]));

  return (
    <div className="vscode">
      <div className="vscode-titlebar">{activeFile} — decider-examples</div>
      <div className="vscode-main">
        <div className="activity-bar">
          <span className={container === "explorer" ? "activity-icon active" : "activity-icon"} title="Explorer" onClick={() => setContainer("explorer")}>▤</span>
          <span className={container === "decider" ? "activity-icon active" : "activity-icon"} title="decider: Pipelines" onClick={() => setContainer("decider")}>◈</span>
          <span className="activity-icon" title="Search" onClick={() => onInert("Search")}>⌕</span>
          <span className="activity-icon" title="Source Control" onClick={() => onInert("Source Control")}>⎇</span>
          <span className="activity-icon" title="Run and Debug" onClick={() => onInert("Run and Debug")}>▷</span>
          <span className="activity-icon" title="Extensions" onClick={() => onInert("Extensions")}>⬡</span>
        </div>
        <div className="sidebar">
          {container === "explorer" ? (
            <>
              <div className="sidebar-title">Explorer</div>
              <ul className="tree">
                <li className="tree-folder">EXAMPLES</li>
                <li className={activeFile === "loan.py" ? "tree-file active" : "tree-file"} onClick={() => onOpenFile("loan.py")}>loan.py</li>
                <li className={activeFile === "largepipeline.py" ? "tree-file active" : "tree-file"} onClick={() => onOpenFile("largepipeline.py")}>largepipeline.py</li>
                <li className="tree-file" onClick={() => onInert("open ladder.py")}>ladder.py</li>
                <li className="tree-file" onClick={() => onInert("open make_bank.py")}>make_bank.py</li>
                <li className="tree-folder" onClick={() => onInert("expand bank/")}>bank</li>
              </ul>
            </>
          ) : (
            <>
              <Pane title="PIPELINES" icon="◈" defaultOpen={!panelOpen}>
                <input
                  className="sidebar-search"
                  type="search"
                  placeholder="Search pipelines…"
                  value={pipelineQuery}
                  onChange={(e) => setPipelineQuery(e.target.value)}
                />
                <ul className="tree">
                  {filteredPipelines.map((p) => (
                    <li key={p.label} className={activeFile === p.file ? "tree-file active" : "tree-file"} onClick={() => onOpenPipeline(p.label)}>
                      {p.label}
                      <span className="tree-file-sub">{p.file}</span>
                    </li>
                  ))}
                  {filteredPipelines.length === 0 && <li className="tree-empty">No pipelines match "{pipelineQuery}"</li>}
                </ul>
              </Pane>
              <Pane title="STRUCTURE" icon="⊞" defaultOpen={panelOpen}>
                {describe ? (
                  <PipelineTree
                    ir={describe.ir}
                    pipeline={describe.pipeline}
                    selected={selectedPath}
                    watches={watches}
                    onSelect={onSelectNode}
                    onToggleDot={onToggleDot}
                    onContextMenu={onDotContextMenu}
                  />
                ) : (
                  <div className="tree-empty">Open a pipeline to see its steps here.</div>
                )}
              </Pane>
              <Pane title="PARAMETERS" icon="⚙" defaultOpen={false}>
                {describe ? (
                  <ul className="tree">
                    <li className="tree-item" onClick={() => onLens("params", "Parameters")}>
                      <span className="tree-twisty" />
                      <span className="tree-label">Open Parameters</span>
                      <span className="tree-desc">
                        {Object.values(describe.params ?? {}).reduce((t, byName) => t + Object.keys(byName).length, 0)} declared
                      </span>
                    </li>
                  </ul>
                ) : (
                  <div className="tree-empty">Visualise a flow to see its parameters.</div>
                )}
              </Pane>
              <Pane title="DATASETS" icon="▦" defaultOpen={panelOpen}>
                <ul className="tree">
                  {DATASETS.map((d) => (
                    <li
                      key={d.name}
                      className={d.name === activeDataset.name ? "tree-item selected" : "tree-item"}
                      title={d.fingerprint ? `fingerprint ${d.fingerprint}` : "no file: rows defined in the module"}
                      onClick={() => onSelectDataset(d.name)}
                    >
                      <span className="tree-twisty" />
                      <span className="tree-label">{d.name}</span>
                      <span className="tree-desc">{d.rowCount.toLocaleString()} rows</span>
                    </li>
                  ))}
                </ul>
              </Pane>
              <Pane title="EXPERIMENTS & RESULTS" icon="⇄" defaultOpen={false}>
                <input
                  className="sidebar-search"
                  type="search"
                  placeholder="Search experiments…"
                  value={experimentQuery}
                  onChange={(e) => setExperimentQuery(e.target.value)}
                />
                <ul className="tree">
                  {filteredExperiments.map((exp) => (
                    <li key={exp.slug} className="tree-file" onClick={() => onSelectExperiment(exp)}>
                      {exp.name}
                    </li>
                  ))}
                  {filteredExperiments.length === 0 && <li className="tree-empty">No experiments match "{experimentQuery}"</li>}
                </ul>
              </Pane>
            </>
          )}
        </div>
        <div className="editor-area">
          <div className={panelOpen && showLeft ? "editor-columns split" : "editor-columns"}>
            {showLeft && (
              <div className="editor-column">
                <div className="tab-bar">
                  {leftTabs.map((tab) => (
                    <span key={tab.id} className={tab.id === activeLeftTabId ? "tab active" : "tab"} onClick={() => onSelectLeftTab(tab.id)}>
                      {tab.kind === "code" ? activeFile : "Parameters"}
                      <button
                        className="tab-close"
                        title="Close"
                        onClick={(e) => {
                          e.stopPropagation();
                          onCloseLeftTab(tab.id);
                        }}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
                {leftTabs.map((tab) => (
                  <div
                    key={tab.id}
                    className={tab.kind === "code" ? "code-view" : "panel-host"}
                    style={{ display: tab.id === activeLeftTabId ? undefined : "none" }}
                  >
                    {tab.kind === "code" ? (
                      file.lines.map((text, i) => {
                        const no = i + 1;
                        const lens = lensByLine.get(no);
                        return (
                          <div key={no}>
                            {lens && (
                              <div className="codelens-row">
                                {lens.buttons.map((b) => (
                                  <button key={b.label} className="codelens-btn" onClick={() => (b.action === "inert" ? onInert(b.label) : onLens(b.action, b.label))}>
                                    {b.label}
                                  </button>
                                ))}
                              </div>
                            )}
                            <div id={`code-line-${no}`} className={revealLine === no ? "code-line code-line-flash" : "code-line"}>
                              <span className="gutter">{no}</span>
                              <span className="code-text">
                                {tokenize(text).map((t, j) => (
                                  <span key={j} className={t.cls}>{t.text}</span>
                                ))}
                              </span>
                            </div>
                          </div>
                        );
                      })
                    ) : describe ? (
                      <ParametersView
                        describe={describe}
                        overrides={paramOverrides}
                        onChange={onChangeParam}
                        onRun={onRunPipeline}
                        focusPath={selectedPath}
                        debugging={debugging}
                        pausedAt={pausedAt}
                        tableRowCounts={Object.fromEntries(
                          Object.entries(tableOverrides).map(([path, byName]) => [path, Object.fromEntries(Object.entries(byName).map(([name, rows]) => [name, rows.length]))]),
                        )}
                        onOpenTable={onOpenTable}
                      />
                    ) : (
                      <div className="tree-empty">Visualise the flow first to see its parameters.</div>
                    )}
                  </div>
                ))}
              </div>
            )}
            {panelOpen && (
              <div className="editor-column">
                <div className="tab-bar">
                  {tabs.map((tab) => (
                    <span key={tab.id} className={tab.id === activeTabId ? "tab active" : "tab"} onClick={() => onSelectTab(tab.id)}>
                      {tab.kind === "flow" ? "decider: flow" : tab.kind === "dataset" ? tab.dataset.name : tab.name}
                      <button
                        className="tab-close"
                        title="Close"
                        onClick={(e) => {
                          e.stopPropagation();
                          onCloseTab(tab.id);
                        }}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
                {tabs.map((tab) => (
                  <div key={tab.id} className="panel-host" style={{ display: tab.id === activeTabId ? undefined : "none" }}>
                    {tab.kind === "flow" ? (
                      panel
                    ) : tab.kind === "dataset" ? (
                      <DatasetView dataset={tab.dataset} onDebugRecord={(row) => onDebugRecord(tab.dataset, row)} />
                    ) : (
                      <ParamTableView
                        path={tab.path}
                        name={tab.name}
                        rows={tableOverrides[tab.path]?.[tab.name] ?? []}
                        onChange={(rows) => onChangeTable(tab.path, tab.name, rows)}
                      />
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
      <div className="vscode-statusbar">
        <span>⎇ main</span>
        <span>Python 3.12.4 ('.venv')</span>
        <span className="statusbar-right">decider</span>
      </div>
    </div>
  );
}
