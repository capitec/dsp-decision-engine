import { useState, type CSSProperties } from "react";
import { callNodes, kindLabel, lastSegment, type IRNodeJson, type Watch } from "../../src/model/protocol";
import { groupPaths } from "../../src/layout";
import { watchText } from "../../src/Controls";

// A native-TreeView-styled stand-in for tools/vscode-decider/src/structure.ts's StructureProvider:
// same icons, same "name -> outputs" shape, same click-to-select-and-reveal behaviour, but as plain
// HTML since there's no real VS Code tree widget outside the extension. Breakpoint dots are a new
// idea being tried here, not something the real sidebar has yet — see notes/vscode-redesign.
// Breakpoint state itself is owned by the host (main.tsx), not this component, so it stays in
// sync with the same dots drawn on the flow panel's graph.

// Mirrors StructureProvider's ICONS map; glyphs stand in for the codicons VS Code would show.
const ICONS: Record<string, string> = {
  scalar: "ƒ", row: "⋔", frame: "▦", sequence: "≡", branch: "⑂", loop: "↻",
};

// Flows up to this many steps start fully expanded, matching App.tsx's own OPEN_ALL threshold.
const OPEN_ALL = 80;

/** The breakpoint at `path`, however it's represented: a plain watch, or a condition scoped to just this node. */
export function breakpointAt(watches: Watch[], path: string): Watch | undefined {
  return watches.find((w) => w.path === path || (w.scope?.length === 1 && w.scope[0] === path));
}

export function PipelineTree({ ir, pipeline, selected, watches, onSelect, onToggleDot, onContextMenu }: {
  ir: IRNodeJson;
  pipeline: string;
  selected?: string;
  watches: Watch[];
  onSelect: (path: string) => void;
  onToggleDot: (path: string) => void;
  onContextMenu: (path: string, rect: DOMRect) => void;
}) {
  const [opened, setOpened] = useState<Set<string>>(() => new Set(callNodes(ir).length <= OPEN_ALL ? groupPaths(ir) : []));

  const toggle = (path: string) =>
    setOpened((o) => {
      const next = new Set(o);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });

  return (
    <div className="pipeline-tree">
      <TreeNode node={ir} pipeline={pipeline} selected={selected} opened={opened} watches={watches} onToggle={toggle} onSelect={onSelect} onToggleDot={onToggleDot} onContextMenu={onContextMenu} depth={0} />
    </div>
  );
}

interface TreeNodeProps {
  node: IRNodeJson;
  pipeline: string;
  selected?: string;
  opened: Set<string>;
  watches: Watch[];
  onToggle: (path: string) => void;
  onSelect: (path: string) => void;
  onToggleDot: (path: string) => void;
  onContextMenu: (path: string, rect: DOMRect) => void;
  depth: number;
}

function TreeNode({ node, pipeline, selected, opened, watches, onToggle, onSelect, onToggleDot, onContextMenu, depth }: TreeNodeProps) {
  const isGroup = node.kind !== "call";
  const open = depth === 0 || opened.has(node.path);
  const call = node.kind === "call" ? node : null;
  const name = node.path === "" ? pipeline : lastSegment(node.path);
  // The root pipeline node has no path of its own to pause on.
  const canBreak = node.path !== "";
  const watch = canBreak ? breakpointAt(watches, node.path) : undefined;

  return (
    <div>
      <div
        className={node.path === selected ? "tree-item selected" : "tree-item"}
        style={{ paddingLeft: 8 + depth * 14 }}
        title={node.path || pipeline}
        onClick={() => (isGroup ? onToggle(node.path) : onSelect(node.path))}
      >
        {isGroup ? <span className="tree-twisty">{open ? "▾" : "▸"}</span> : <span className="tree-twisty" />}
        {canBreak && (
          <span
            className={watch ? "bp-dot bp-set" : "bp-dot"}
            title={watch ? `Breakpoint: ${watchText(watch, null)} — click to remove, right-click to edit` : "Click to set a breakpoint, right-click for more"}
            onClick={(e) => {
              e.stopPropagation();
              onToggleDot(node.path);
            }}
            onContextMenu={(e) => {
              e.preventDefault();
              e.stopPropagation();
              onContextMenu(node.path, e.currentTarget.getBoundingClientRect());
            }}
          >
            {(watch?.name !== undefined || watch?.condition !== undefined) && <span className="bp-eq">=</span>}
          </span>
        )}
        <span className="tree-icon">{ICONS[kindLabel(node)] ?? "•"}</span>
        <span className="tree-label">{name}</span>
        {call && <span className="tree-desc">→ {call.outputs?.join(", ") || "?"}</span>}
      </div>
      {isGroup && open && node.children.map((c) => (
        <TreeNode key={c.path} node={c} pipeline={pipeline} selected={selected} opened={opened} watches={watches} onToggle={onToggle} onSelect={onSelect} onToggleDot={onToggleDot} onContextMenu={onContextMenu} depth={depth + 1} />
      ))}
    </div>
  );
}

/** A small right-click menu, shared by the sidebar's dot and the graph's node. */
export function ContextMenu({ style, items }: { style: CSSProperties; items: { label: string; onClick: () => void }[] }) {
  return (
    <div className="bp-menu" style={{ ...style, position: "fixed" }}>
      {items.map((item) => (
        <button key={item.label} onClick={item.onClick}>{item.label}</button>
      ))}
    </div>
  );
}

export function BreakpointEditor({ style, label, inputs, watch, onSave, onRemove, onCancel }: {
  style: CSSProperties;
  label: string;
  inputs: string[];
  watch?: Watch;
  onSave: (expression: string) => void;
  onRemove: () => void;
  onCancel: () => void;
}) {
  const initial = watch?.condition ?? (watch?.name !== undefined ? `${watch.name} ${watch.op} ${JSON.stringify(watch.value)}` : "");
  const [expression, setExpression] = useState(initial);

  return (
    <div className="bp-editor" style={{ ...style, position: "fixed" }} onClick={(e) => e.stopPropagation()}>
      <div className="bp-editor-title">Edit breakpoint · {label}</div>
      {watch && <div className="bp-editor-current">Currently: {watchText(watch, null)}</div>}
      <div className="bp-editor-row">
        <input
          aria-label="break when expression"
          className="bp-expression"
          placeholder="Expression, e.g. min_net_salary < 5000"
          autoFocus
          value={expression}
          onChange={(e) => setExpression(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && expression.trim() !== "" && onSave(expression)}
        />
      </div>
      {inputs.length > 0 && <div className="bp-editor-hint">Inputs here: {inputs.join(", ")}</div>}
      <div className="bp-editor-row">
        <button onClick={() => onSave(expression)} disabled={expression.trim() === ""}>Save</button>
        <button onClick={onRemove}>Remove</button>
        <button onClick={onCancel}>Cancel</button>
      </div>
    </div>
  );
}
