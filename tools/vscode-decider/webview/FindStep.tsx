import { useEffect, useRef, useState } from "react";
import type { CallNodeJson } from "../src/protocol";

const MAX_HITS = 12;

/** Steps whose name, description or group matches every word of `query`, best first. */
export function findSteps(nodes: CallNodeJson[], query: string): CallNodeJson[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  const text = (n: CallNodeJson) => `${n.path} ${n.doc ?? ""}`.toLowerCase().replace(/_/g, " ");
  const name = (n: CallNodeJson) => n.path.slice(n.path.lastIndexOf("/") + 1).toLowerCase().replace(/_/g, " ");
  return nodes
    .filter((n) => words.every((w) => text(n).includes(w.replace(/_/g, " "))))
    .sort((a, b) => Number(words.every((w) => name(b).includes(w))) - Number(words.every((w) => name(a).includes(w))));
}

/** A find box over every step; picking a hit selects it (and opens the groups around it). Ctrl+F focuses it. */
export function FindStep({ nodes, onPick }: { nodes: CallNodeJson[]; onPick: (path: string) => void }) {
  const [query, setQuery] = useState("");
  const [at, setAt] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const hits = findSteps(nodes, query);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "f") {
        e.preventDefault();
        input.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const pick = (n: CallNodeJson | undefined) => {
    if (!n) return;
    onPick(n.path);
    setQuery("");
  };
  return (
    <span className="find">
      <input
        ref={input}
        aria-label="Find a step"
        placeholder={`Find a step among ${nodes.length} (Ctrl+F)`}
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setAt(0);
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") setAt(Math.min(at + 1, Math.min(hits.length, MAX_HITS) - 1));
          else if (e.key === "ArrowUp") setAt(Math.max(at - 1, 0));
          else if (e.key === "Enter") pick(hits[at]);
          else if (e.key === "Escape") setQuery("");
        }}
      />
      {query && (
        <ul className="find-hits" role="listbox">
          {hits.slice(0, MAX_HITS).map((n, i) => (
            <li key={n.path} role="option" aria-selected={i === at} className={i === at ? "at" : ""} onMouseDown={() => pick(n)}>
              <strong>{n.path.slice(n.path.lastIndexOf("/") + 1)}</strong>
              {n.doc && <span> · {n.doc}</span>}
              <div className="muted small">in {n.path.slice(0, n.path.lastIndexOf("/"))}</div>
            </li>
          ))}
          {hits.length === 0 && <li className="muted">No step matches “{query}”.</li>}
          {hits.length > MAX_HITS && <li className="muted small">{hits.length - MAX_HITS} more: type more to narrow</li>}
        </ul>
      )}
    </span>
  );
}
