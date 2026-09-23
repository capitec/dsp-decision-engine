import { useMemo } from "react";
import type { IRNodeJson } from "../src/protocol";
import { layout } from "./layout";

interface Props {
  ir: IRNodeJson;
  showData: boolean;
  current?: string;
  finished: string[];
  selected?: string;
  highlightColumn?: string;
  onSelect: (path: string) => void;
  onOpen: (path: string) => void;
}

export function Graph({ ir, showData, current, finished, selected, highlightColumn, onSelect, onOpen }: Props) {
  const laid = useMemo(() => layout(ir, showData), [ir, showData]);
  const done = new Set(finished);
  const touches = (n: IRNodeJson) => n.kind === "call" && !!highlightColumn && (n.inputs.includes(highlightColumn) || n.outputs.includes(highlightColumn));

  return (
    <div className="graph">
      <svg width={laid.width} height={laid.height}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--vscode-foreground)" />
          </marker>
        </defs>
        {laid.clusters.map((c) => (
          <g key={c.path} className={`cluster ${c.kind}`}>
            <rect x={c.x} y={c.y} width={c.width} height={c.height} rx={6} />
            <text x={c.x + 8} y={c.y + 14}>{c.label} · {c.kind}</text>
          </g>
        ))}
        {laid.edges.map((e) => {
          const related = selected && (e.from === selected || e.to === selected);
          const onColumn = highlightColumn && e.kind === "data" && e.label === highlightColumn;
          const d = e.points.map((p, i) => `${i ? "L" : "M"}${p.x},${p.y}`).join(" ");
          const mid = e.points[Math.floor(e.points.length / 2)];
          return (
            <g key={e.id} className={`edge ${e.kind} ${related ? "related" : ""} ${onColumn ? "column" : ""}`}>
              <path d={d} markerEnd="url(#arrow)" />
              {e.label && (
                <text x={mid.x} y={mid.y - 3} textAnchor="middle">{e.label}</text>
              )}
            </g>
          );
        })}
        {laid.nodes.map((n) => (
          <g
            key={n.path}
            className={`node ${n.path === current ? "current" : ""} ${done.has(n.path) ? "done" : ""} ${n.path === selected ? "selected" : ""} ${touches(n.node) ? "touches" : ""}`}
            transform={`translate(${n.x},${n.y})`}
            onClick={() => onSelect(n.path)}
            onDoubleClick={() => onOpen(n.path)}
          >
            <rect width={n.width} height={n.height} rx={5} />
            <text x={n.width / 2} y={18} textAnchor="middle" className="title">{n.label}</text>
            <text x={n.width / 2} y={34} textAnchor="middle" className="sub">
              {n.node.kind === "call" ? `→ ${n.node.outputs.join(", ")}` : ""}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}
