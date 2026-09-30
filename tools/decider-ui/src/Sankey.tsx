import { sankeyLayout, type Sankey } from "./model/experiment";

/** A Sankey of record flow through branches and into outcomes, drawn as SVG. */
export function SankeyDiagram({ sankey }: { sankey: Sankey }) {
  const { nodes, links, width, height } = sankeyLayout(sankey);
  const linkPath = (l: (typeof links)[number]) => {
    const mx = (l.x1 + l.x2) / 2;
    return `M ${l.x1} ${l.y1} C ${mx} ${l.y1}, ${mx} ${l.y2}, ${l.x2} ${l.y2}`;
  };
  return (
    <svg className="sankey" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Records flowing through branches into ${sankey.outcome}`}>
      <g>
        {links.map((l) => (
          <path key={`${l.source}->${l.target}`} className={`link ${l.source === "in" ? "" : "to-outcome"}`} d={linkPath(l)} strokeWidth={Math.max(1, l.value * 6)}>
            <title>{`${l.value} record${l.value === 1 ? "" : "s"}`}</title>
          </path>
        ))}
      </g>
      <g>
        {nodes.map((n) => (
          <g key={n.id} className={`node ${n.kind}`} transform={`translate(${n.x}, ${n.y})`}>
            <rect width={n.w} height={n.h} rx={4} />
            <text x={n.w / 2} y={n.h / 2} dominantBaseline="middle" textAnchor="middle">{n.label}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}
