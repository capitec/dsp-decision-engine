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
  /** The focused record's path through a tree step, shown on that step. */
  treePath?: { path: string; visited: string[] } | null;
  /** "auto" fits the panel's width without shrinking text below MIN_AUTO; a number is a fixed scale. */
  zoom: number | "auto";
  onSelect: (path: string) => void;
  onOpen: (path: string) => void;
  /** Open a folded group, or fold an open one. */
  onToggle: (path: string) => void;
}

// A step whose output many later steps read would bury the graph in arcs; past this, only its inputs are drawn.
const MAX_OUT = 4;

// Below this, node text gets smaller than the editor's; scroll instead.
const MIN_AUTO = 0.9;

export function Graph({ ir, showData, run, selected, highlightColumn, lineage, diff, treePath, zoom, onSelect, onOpen, onToggle }: Props) {
  const laid = useMemo(() => layout(ir), [ir]);
  const flows = useMemo(() => dataEdges(ir), [ir]);
  const at = useMemo(() => new Map(laid.nodes.map((n) => [n.path, n])), [laid]);
  // Data edges would tangle the layout, so they are drawn over it: all of them, or only the selection's.
  const outgoing = flows.filter((f) => f.from === selected).length;
  const shownFlows = flows.filter((f) => showData || f.to === selected || (f.from === selected && outgoing <= MAX_OUT) || f.column === highlightColumn);
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const watch = new ResizeObserver(() => setWidth(el.clientWidth));
    watch.observe(el);
    return () => watch.disconnect();
  }, []);
  // A new flow starts at its top.
  useEffect(() => box.current?.scrollTo(0, 0), [ir.path]);
  const done = new Set(run.finishedPaths);
  const touches = (n: IRNodeJson) =>
    n.kind === "call" && !!highlightColumn && ((n.inputs ?? []).includes(highlightColumn) || (n.outputs ?? []).includes(highlightColumn));
  const lines = (n: IRNodeJson): [string, string] => {
    if (n.kind !== "call") return n.folded ? foldedLines(n.folded) : ["", ""];
    if (treePath?.path === n.path && n.source.includes("DecisionTable")) return [`row ${Number(treePath.visited[treePath.visited.length - 1]) + 1} matched`, "for the focused record"];
    if (treePath?.path === n.path) return [fit(treePath.visited.slice(1).join(" → ")), "path of the focused record"];
    return [fit(`reads ${n.inputs === null ? "?" : n.inputs.join(", ") || "nothing"}`), fit(`→ ${n.outputs === null ? "?" : n.outputs.join(", ")}`)];
  };

  // Keep the step the run is at, or the one just clicked, in the middle of the view.
  useEffect(() => {
    box.current?.querySelector(".node.current, .cluster.current")?.scrollIntoView({ block: "center", inline: "nearest", behavior: "smooth" });
  }, [run.current?.path, zoom]);
  useEffect(() => {
    box.current?.querySelector(".node.selected")?.scrollIntoView({ block: "center", inline: "center", behavior: "smooth" });
  }, [selected]);

  const pad = shownFlows.length ? PAD : 0;

  const full = laid.width + pad;
  const scale = zoom === "auto" ? Math.min(1, Math.max(MIN_AUTO, width ? (width - 4) / full : 1)) : zoom;

  return (
    <div className="graph" ref={box}>
      <svg viewBox={`0 0 ${full} ${laid.height}`} width={full * scale} height={laid.height * scale}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--vscode-foreground)" />
          </marker>
        </defs>
        {laid.clusters.filter((c) => c.path !== ir.path).map((c) => (
          <g key={c.path} className={`cluster ${c.kind} ${run.current?.path === c.path ? "current" : ""}`}>
            <rect x={c.x} y={c.y} width={c.width} height={c.height} rx={6} />
            <text x={c.x + 8} y={c.y + 15} className="fold" onClick={() => onToggle(c.path)}>
              <title>Fold {c.label} into one box</title>⊟ {c.label} <tspan className="kind">{c.kind}</tspan>
            </text>
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
              n.node.kind !== "call" ? "folded" : "",
              run.edits?.[n.path] ? `edited-${run.edits[n.path]}` : "",
              n.node.kind !== "call" && run.current && n.node.folded?.includes(run.current.path) ? "current" : "",
              n.path === run.current?.path ? `current ${run.current.when}` : "",
              done.has(n.path) ? "done" : "",
              n.path === selected ? "selected" : "",
              touches(n.node) ? "touches" : "",
              lineage.has(n.path) ? "lineage" : "",
              diff ? `diff-${(diff.get(n.path) ?? "same").replace(" ", "-")}` : "",
            ].join(" ")}
            transform={`translate(${n.x},${n.y})`}
            onClick={() => (n.node.kind === "call" ? onSelect(n.path) : onToggle(n.path))}
            onDoubleClick={() => n.node.kind === "call" && onOpen(n.path)}
          >
            <title>
              {n.node.kind === "call"
                ? `${n.path} (${kindLabel(n.node)})${n.node.doc ? `\n${n.node.doc}` : ""}\n${(n.node.inputs ?? ["?"]).join(", ")} → ${(n.node.outputs ?? ["?"]).join(", ")}\nDouble-click to open the source`
                : `${n.path}: ${n.node.folded?.length} steps. Click to open.`}
            </title>
            <rect width={n.width} height={n.height} rx={5} />
            <text x={n.width / 2} y={19} textAnchor="middle" className="title">
              {n.node.kind === "call" && n.node.table ? "▦ " : n.node.kind === "call" && n.node.callKind === "row" ? "◇ " : n.node.kind === "call" && n.node.callKind === "frame" ? "⊞ " : n.node.kind !== "call" ? "⊞ " : ""}
              {n.label}
            </text>
            <text x={n.width / 2} y={35} textAnchor="middle" className="sub">{lines(n.node)[0]}</text>
            <text x={n.width / 2} y={50} textAnchor="middle" className="sub">{lines(n.node)[1]}</text>
            {run.edits?.[n.path] && (
              <text x={n.width - 6} y={13} textAnchor="end" className="edit-mark">{run.edits[n.path] === "delete" ? "skipped" : "edited"}</text>
            )}
            {done.has(n.path) && (
              <text x={6} y={13} className="ran-mark"><title>ran</title>✓</text>
            )}
          </g>
        ))}
      </svg>
    </div>
  );

/** A data edge: out of the right side of the writer, into the right side of the reader. */
function curve(a: LaidNode, b: LaidNode): string {
  const [x1, y1] = [a.x + a.width, a.y + a.height / 2];
  const [x2, y2] = [b.x + b.width, b.y + b.height / 2];
  const bulge = 30 + Math.min(120, Math.abs(y2 - y1) / 4);
  return `M${x1},${y1} C${x1 + bulge},${y1} ${x2 + bulge},${y2} ${x2},${y2}`;
}

  function foldedLines(paths: string[]): [string, string] {
    const changed = diff ? paths.filter((p) => diff.get(p) === "changed" || diff.get(p) === "added").length : 0;
    const ran = paths.filter((p) => done.has(p)).length;
    const status = changed ? `${changed} changed` : ran ? `${ran} of ${paths.length} ran` : "";
    return [`${paths.length} steps · click to open`, status];
  }
}

/** Shortened to fit on a node; the node's tooltip has the whole text. */
function fit(text: string): string {
  return text.length > 36 ? `${text.slice(0, 35)}…` : text;
}
