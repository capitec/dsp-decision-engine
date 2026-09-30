import { useEffect, useMemo, useRef, useState } from "react";
import { kindLabel, type CallNodeJson, type IRNodeJson, type RunStatus } from "./model/protocol";
import type { BreakpointKind } from "./model/breakpoints";
import { dataEdges, fold, layout, type LaidNode, type Layout } from "./layout";
import type { Selection } from "./Inspector";

const MIN_K = 0.1;
const MAX_K = 4;
const FIT_PAD = 40;

interface Props {
  ir: IRNodeJson;
  open: (path: string) => boolean;
  selected?: Selection;
  onSelect: (s: Selection | undefined) => void;
  onToggle: (path: string) => void;
  onReveal?: (path: string) => void;
  nodes: CallNodeJson[];
  run?: RunStatus;
  breakpoints?: Map<string, BreakpointKind>;
}

export function Graph({ ir, open, selected, onSelect, onToggle, onReveal, nodes, run, breakpoints }: Props) {
  const shown = useMemo(() => fold(ir, open), [ir, open]);
  const laid = useMemo(() => layout(shown), [shown]);
  const flows = useMemo(() => dataEdges(shown), [shown]);
  const at = useMemo(() => new Map(laid.nodes.map((n) => [n.path, n])), [laid]);

  const box = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [view, setView] = useState({ x: FIT_PAD, y: FIT_PAD, k: 1 });
  const fittedFor = useRef<string>();

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const watch = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    watch.observe(el);
    return () => watch.disconnect();
  }, []);

  // Keep the selected node's screen position through a re-layout (fold/unfold, re-describe).
  const prevLaid = useRef<Layout>();
  const selectedPath = selected?.kind === "node" ? selected.path : undefined;
  useEffect(() => {
    const old = prevLaid.current;
    prevLaid.current = laid;
    if (!old || !selectedPath) return;
    const before = old.nodes.find((n) => n.path === selectedPath);
    const after = laid.nodes.find((n) => n.path === selectedPath);
    if (!before || !after) return;
    const sx = (before.x + before.width / 2) * view.k + view.x;
    const sy = (before.y + before.height / 2) * view.k + view.y;
    setView((v) => ({ ...v, x: sx - (after.x + after.width / 2) * v.k, y: sy - (after.y + after.height / 2) * v.k }));
  }, [laid, selectedPath]);

  // Fit a flow once it lays out, when the container has a size; not on later size changes (that
  // would fight a user's zoom/pan). Re-fits once per flow, keyed by the flow's identity.
  useEffect(() => {
    if (size.w === 0 || laid.width === 0) return;
    if (fittedFor.current === ir.path) return;
    fittedFor.current = ir.path;
    const k = Math.max(MIN_K, Math.min(1, (size.w - FIT_PAD * 2) / laid.width, (size.h - FIT_PAD * 2) / laid.height));
    setView({ k, x: (size.w - laid.width * k) / 2, y: (size.h - laid.height * k) / 2 });
  }, [laid, size, ir.path]);

  // The selected value's touch points, for highlighting.
  const valueName = selected?.kind === "value" ? selected.name : undefined;
  const shownFlows = flows.filter((f) =>
    selected?.kind === "edge" && f.from === selected.from && f.to === selected.to
      ? true
      : selected?.kind === "node"
        ? f.from === selected.path || f.to === selected.path
        : selected?.kind === "value"
          ? f.column === selected.name
          : false,
  );
  const selectedEdge = selected?.kind === "edge" ? selected : undefined;
  const touches = (n: IRNodeJson) => n.kind === "call" && valueName !== undefined && ((n.inputs ?? []).includes(valueName) || (n.outputs ?? []).includes(valueName));

  // ---- gestures ------------------------------------------------------------
  const zoomAt = (cx: number, cy: number, factor: number) =>
    setView((v) => {
      const k = clamp(v.k * factor);
      const r = k / v.k;
      return { k, x: cx - (cx - v.x) * r, y: cy - (cy - v.y) * r };
    });

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      zoomAt(e.clientX - rect.left, e.clientY - rect.top, e.deltaY < 0 ? 1.1 : 1 / 1.1);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const drag = useRef<{ x: number; y: number; ox: number; oy: number; moved: boolean } | null>(null);
  const panStart = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    drag.current = { x: e.clientX, y: e.clientY, ox: view.x, oy: view.y, moved: false };
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
  };
  const panMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
    setView((v) => ({ ...v, x: d.ox + dx, y: d.oy + dy }));
  };
  const panEnd = (e: React.PointerEvent) => {
    const d = drag.current;
    drag.current = null;
    if (d && !d.moved) onSelect(undefined); // a plain click on empty canvas clears the selection
  };

  const fit = () => {
    const k = Math.max(MIN_K, Math.min(1, (size.w - FIT_PAD * 2) / laid.width, (size.h - FIT_PAD * 2) / laid.height));
    setView({ k, x: (size.w - laid.width * k) / 2, y: (size.h - laid.height * k) / 2 });
  };
  const fitSelection = () => {
    const n = selectedPath && laid.nodes.find((x) => x.path === selectedPath);
    if (!n) return;
    setView({ k: 1, x: size.w / 2 - (n.x + n.width / 2), y: size.h / 2 - (n.y + n.height / 2) });
  };

  return (
    <div className="graph" ref={box}>
      <svg onPointerDown={panStart} onPointerMove={panMove} onPointerUp={panEnd} onPointerLeave={panEnd}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--decider-fg)" />
          </marker>
        </defs>
        <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}>
          {laid.clusters.filter((c) => c.path !== ir.path).map((c) => (
            <g key={c.path} className={`cluster ${c.kind}`}>
              <rect x={c.x} y={c.y} width={c.width} height={c.height} rx={8} />
              <text x={c.x + 10} y={c.y + 16} className="fold" onPointerDown={(e) => e.stopPropagation()} onClick={() => onToggle(c.path)}>
                ⊟ {c.label} <tspan className="kind">{c.kind}</tspan>
              </text>
            </g>
          ))}
          {laid.edges.map((e) => (
            <Edge key={e.id} e={e} selected={selectedEdge?.id === e.id} onPick={() => onSelect({ kind: "edge", id: e.id, from: e.from, to: e.to, label: e.label })} />
          ))}
          {shownFlows.map((f, i) => {
            const a = at.get(f.from);
            const b = at.get(f.to);
            if (!a || !b) return null;
            const id = `d${i}`;
            const isSelected = selectedEdge?.id === id;
            return (
              <DataEdge
                key={id}
                a={a}
                b={b}
                column={f.column}
                selected={isSelected}
                onPick={() => onSelect({ kind: "edge", id, from: f.from, to: f.to, columns: [f.column] })}
              />
            );
          })}
          {laid.nodes.map((n) => {
            const node = n.node;
            const call = node.kind === "call" ? node : null;
            const bk = breakpoints?.get(n.path);
            const paused = run?.current?.path === n.path;
            const ran = run ? run.finishedPaths.includes(n.path) : false;
            return (
              <g
                key={n.path}
                className={[
                  "node",
                  call ? kindLabel(call) : "folded",
                  n.path === selectedPath ? "selected" : "",
                  touches(node) ? "touches" : "",
                  bk ? "breakpoint" : "",
                  paused ? "paused" : "",
                  ran && !paused ? "ran" : "",
                ].join(" ")}
                transform={`translate(${n.x},${n.y})`}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={() => (call ? onSelect({ kind: "node", path: n.path }) : onToggle(n.path))}
                onDoubleClick={() => call && onReveal?.(n.path)}
              >
                <title>
                  {call
                    ? `${n.path} (${kindLabel(call)})${call.doc ? `\n${call.doc}` : ""}\n${(call.inputs ?? ["?"]).join(", ")} → ${(call.outputs ?? ["?"]).join(", ")}\nDouble-click to open the source`
                    : `${n.path}: ${node.kind === "call" ? "" : node.folded?.length} steps. Click to open.`}
                  {bk ? `\nBreakpoint: ${bk}` : ""}{paused ? `\nPaused ${run!.current!.when}` : ran ? "\nHas run" : ""}
                </title>
                <rect width={n.width} height={n.height} rx={6} />
                <text x={n.width / 2} y={20} textAnchor="middle" className="title">{icon(node)}{n.label}</text>
                <text x={n.width / 2} y={37} textAnchor="middle" className="sub">{subLine(node)[0]}</text>
                <text x={n.width / 2} y={51} textAnchor="middle" className="sub">{subLine(node)[1]}</text>
                {bk && <circle className="bp-dot" cx={n.width - 8} cy={8} r={5} />}
                {paused && <circle className="pause-dot" cx={8} cy={8} r={5} />}
                {ran && !paused && <circle className="ran-dot" cx={8} cy={n.height - 8} r={3.5} />}
              </g>
            );
          })}
        </g>
      </svg>
      <div className="graph-tools">
        <button title="Zoom out" onClick={() => zoomAt(size.w / 2, size.h / 2, 1 / 1.25)}>−</button>
        <button title="Fit the whole flow" onClick={fit}>fit</button>
        <button title="Zoom in" onClick={() => zoomAt(size.w / 2, size.h / 2, 1.25)}>+</button>
        {selectedPath && <button title="Fit the selected step" onClick={fitSelection}>focus</button>}
      </div>
      <div className="graph-hint">wheel to zoom · drag to pan · click a step or edge to inspect</div>
    </div>
  );

  function subLine(n: IRNodeJson): [string, string] {
    if (n.kind !== "call") return n.folded ? [`${n.folded.length} steps`, "click to open"] : ["", ""];
    return [truncate(`reads ${n.inputs === null ? "?" : n.inputs.join(", ") || "nothing"}`), truncate(`→ ${n.outputs === null ? "?" : n.outputs.join(", ")}`)];
  }
}

function Edge({ e, selected, onPick }: { e: Layout["edges"][number]; selected: boolean; onPick: () => void }) {
  const d = e.points.map((p, i) => `${i ? "L" : "M"}${p.x},${p.y}`).join(" ");
  const mid = e.points[Math.floor(e.points.length / 2)];
  return (
    <g className={`edge ${e.kind} ${selected ? "selected" : ""}`}>
      <path d={d} markerEnd="url(#arrow)" />
      <path d={d} className="hit" onPointerDown={(ev) => ev.stopPropagation()} onClick={onPick} />
      {e.label && <text x={mid.x} y={mid.y - 4} textAnchor="middle">{e.label}</text>}
    </g>
  );
}

function DataEdge({ a, b, column, selected, onPick }: { a: LaidNode; b: LaidNode; column: string; selected: boolean; onPick: () => void }) {
  const d = curve(a, b);
  const mid = { x: (a.x + a.width + b.x + b.width) / 2, y: (a.y + a.height + b.y) / 2 };
  return (
    <g className={`edge data ${selected ? "selected" : ""}`}>
      <path d={d} markerEnd="url(#arrow)" />
      <path d={d} className="hit" onPointerDown={(ev) => ev.stopPropagation()} onClick={onPick} />
      <text x={mid.x + 4} y={mid.y}>{column}</text>
    </g>
  );
}

/** A data edge: out of the right side of the writer, into the right side of the reader. */
function curve(a: LaidNode, b: LaidNode): string {
  const [x1, y1] = [a.x + a.width, a.y + a.height / 2];
  const [x2, y2] = [b.x + b.width, b.y + b.height / 2];
  const bulge = 30 + Math.min(120, Math.abs(y2 - y1) / 4);
  return `M${x1},${y1} C${x1 + bulge},${y1} ${x2 + bulge},${y2} ${x2},${y2}`;
}

const icon = (n: IRNodeJson) => (n.kind !== "call" ? "⊞ " : n.table ? "▦ " : n.callKind === "row" ? "◇ " : n.callKind === "frame" ? "⊞ " : "");

/** Shortened to fit on a node; the node's tooltip has the whole text. */
function truncate(text: string): string {
  return text.length > 38 ? `${text.slice(0, 37)}…` : text;
}

function clamp(k: number): number {
  return Math.min(MAX_K, Math.max(MIN_K, k));
}
