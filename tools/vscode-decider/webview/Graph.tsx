import { useEffect, useMemo, useRef, useState } from "react";
import { kindLabel, type IRNodeJson, type RunStatus } from "../src/protocol";
import { dataEdges, layout, type LaidNode } from "./layout";

// Room on the right for data edges, which curve out past the nodes.
const PAD = 140;

interface Props {
  ir: IRNodeJson;
  showData: boolean;
  run: RunStatus;
  selected?: string;
  highlightColumn?: string;
  lineage: Set<string>;
  /** Per-step status from a comparison, to colour the graph. */
  diff?: Map<string, string>;
  onSelect: (path: string) => void;
  onOpen: (path: string) => void;
}

export function Graph({ ir, showData, run, selected, highlightColumn, lineage, diff, onSelect, onOpen }: Props) {
  const laid = useMemo(() => layout(ir), [ir]);
  const flows = useMemo(() => dataEdges(ir), [ir]);
  const at = useMemo(() => new Map(laid.nodes.map((n) => [n.path, n])), [laid]);
  // Data edges would tangle the layout, so they are drawn over it: all of them, or only the selection's.
  const shownFlows = flows.filter((f) => showData || f.from === selected || f.to === selected || f.column === highlightColumn);
  const [zoom, setZoom] = useState<number | "fit">("fit");
  const box = useRef<HTMLDivElement>(null);
  const done = new Set(run.finishedPaths);
  const touches = (n: IRNodeJson) =>
    n.kind === "call" && !!highlightColumn && ((n.inputs ?? []).includes(highlightColumn) || (n.outputs ?? []).includes(highlightColumn));
  const subtitle = (n: IRNodeJson) => {
    if (n.kind !== "call") return "";
    const visits = run.visits[n.path];
    if (visits && Object.keys(visits).length) return Object.entries(visits).map(([l, c]) => `#${l} ${c}`).join(" · ");
    return n.outputs === null ? "→ ?" : `→ ${n.outputs.join(", ")}`;
  };

  // Keep the step the run is at in view.
  useEffect(() => {
    box.current?.querySelector(".node.current, .cluster.current")?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
  }, [run.current?.path, zoom]);

  const scale = zoom === "fit" ? 1 : zoom;
  const step = (f: number) => setZoom((z) => Math.min(3, Math.max(0.3, (z === "fit" ? 1 : z) * f)));

  return (
    <div className="graph" ref={box}>
      <div className="zoom">
        <button title="Zoom out" onClick={() => step(1 / 1.25)}>−</button>
        <button title="Fit to width" className={zoom === "fit" ? "active" : ""} onClick={() => setZoom("fit")}>fit</button>
        <button title="Zoom in" onClick={() => step(1.25)}>+</button>
      </div>
      <svg
        viewBox={`0 0 ${laid.width + PAD} ${laid.height}`}
        width={zoom === "fit" ? "100%" : (laid.width + PAD) * scale}
        height={zoom === "fit" ? undefined : laid.height * scale}
        style={zoom === "fit" ? { maxWidth: (laid.width + PAD) * 1.5 } : undefined}
      >
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--vscode-foreground)" />
          </marker>
        </defs>
        {laid.clusters.map((c) => (
          <g key={c.path} className={`cluster ${c.kind} ${run.current?.path === c.path ? "current" : ""}`} onClick={() => onSelect(c.path)}>
            <rect x={c.x} y={c.y} width={c.width} height={c.height} rx={6} />
            <text x={c.x + 8} y={c.y + 14}>{c.label} · {c.kind}</text>
          </g>
        ))}
        {laid.edges.map((e) => {
          const related = selected && (e.from === selected || e.to === selected);
          const d = e.points.map((p, i) => `${i ? "L" : "M"}${p.x},${p.y}`).join(" ");
          const mid = e.points[Math.floor(e.points.length / 2)];
          return (
            <g key={e.id} className={`edge ${e.kind} ${related ? "related" : ""}`}>
              <path d={d} markerEnd="url(#arrow)" />
              {e.label && <text x={mid.x} y={mid.y - 3} textAnchor="middle">{e.label}</text>}
            </g>
          );
        })}
        {shownFlows.map((f, i) => {
          const a = at.get(f.from);
          const b = at.get(f.to);
          if (!a || !b) return null;
          const onColumn = f.column === highlightColumn;
          return (
            <g key={`f${i}`} className={`edge data ${onColumn ? "column" : ""} ${f.from === selected || f.to === selected ? "related" : ""}`}>
              <path d={curve(a, b)} markerEnd="url(#arrow)" />
              <text x={(a.x + b.x + a.width) / 2 + 4} y={(a.y + a.height + b.y) / 2}>{f.column}</text>
            </g>
          );
        })}
        {laid.nodes.map((n) => (
          <g
            key={n.path}
            className={[
              "node",
              kindLabel(n.node),
              n.path === run.current?.path ? `current ${run.current.when}` : "",
              done.has(n.path) ? "done" : "",
              n.path === selected ? "selected" : "",
              touches(n.node) ? "touches" : "",
              lineage.has(n.path) ? "lineage" : "",
              diff ? `diff-${(diff.get(n.path) ?? "same").replace(" ", "-")}` : "",
            ].join(" ")}
            transform={`translate(${n.x},${n.y})`}
            onClick={() => onSelect(n.path)}
            onDoubleClick={() => onOpen(n.path)}
          >
            <title>{`${n.path}\n${kindLabel(n.node)} · double-click to open the source`}</title>
            <rect width={n.width} height={n.height} rx={5} />
            <text x={n.width / 2} y={18} textAnchor="middle" className="title">{n.label}</text>
            <text x={n.width / 2} y={34} textAnchor="middle" className="sub">{subtitle(n.node)}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

/** A data edge: out of the right side of the writer, into the right side of the reader. */
function curve(a: LaidNode, b: LaidNode): string {
  const [x1, y1] = [a.x + a.width, a.y + a.height / 2];
  const [x2, y2] = [b.x + b.width, b.y + b.height / 2];
  const bulge = 30 + Math.min(120, Math.abs(y2 - y1) / 4);
  return `M${x1},${y1} C${x1 + bulge},${y1} ${x2 + bulge},${y2} ${x2},${y2}`;
}
