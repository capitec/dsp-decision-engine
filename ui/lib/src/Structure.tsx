import { lastSegment, type IRNodeJson } from "./model/protocol";

/** The flow as a tree: the same open/selected state as the graph, so the two stay in step. */
export function Structure({ ir, selected, opened, onSelect, onToggle }: {
  ir: IRNodeJson;
  selected?: string;
  opened: Set<string>;
  onSelect: (path: string) => void;
  onToggle: (path: string) => void;
}) {
  return (
    <nav className="structure" aria-label="Structure">
      <Tree node={ir} selected={selected} opened={opened} onSelect={onSelect} onToggle={onToggle} depth={0} />
    </nav>
  );
}

function Tree({ node, selected, opened, onSelect, onToggle, depth }: {
  node: IRNodeJson;
  selected?: string;
  opened: Set<string>;
  onSelect: (path: string) => void;
  onToggle: (path: string) => void;
  depth: number;
}) {
  const isGroup = node.kind !== "call";
  const open = isGroup && (depth === 0 || opened.has(node.path) || selected?.startsWith(`${node.path}/`));
  const call = node.kind === "call" ? node : null;
  return (
    <div>
      <button
        className={`tree-node ${node.path === selected ? "selected" : ""}`}
        style={{ paddingLeft: 12 + depth * 14 }}
        onClick={() => (isGroup ? onToggle(node.path) : onSelect(node.path))}
        title={node.path}
      >
        {isGroup && <span className="twisty">{open ? "▾" : "▸"}</span>}
        <span className={`kind kind-${node.kind === "call" ? call!.callKind : node.kind}`}>{node.kind === "call" ? call!.callKind[0].toUpperCase() : node.kind[0].toUpperCase()}</span>
        <span className="name">{lastSegment(node.path)}</span>
        {call && <span className="writes">{call.outputs?.join(", ") || "?"}</span>}
      </button>
      {isGroup && open && node.children.map((c) => (
        <Tree key={c.path} node={c} selected={selected} opened={opened} onSelect={onSelect} onToggle={onToggle} depth={depth + 1} />
      ))}
    </div>
  );
}
