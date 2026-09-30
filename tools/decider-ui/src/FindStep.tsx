import { useEffect, useRef, useState } from "react";
import type { CallNodeJson } from "./model/protocol";

const MAX_HITS = 12;

/** The branch arm a step sits in (e.g. "personal_loan"), which tells same-named rules apart. */
const tagOf = (path: string) => {
  const parts = path.split("/");
  const i = parts.indexOf("product");
  return i >= 0 && parts[i + 1] && i + 1 < parts.length - 1 ? parts[i + 1] : "";
};

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
export function FindStep({ nodes, onPick, selected }: { nodes: CallNodeJson[]; onPick: (path: string) => void; selected?: string }) {
  const [query, setQuery] = useState("");
  const [at, setAt] = useState(0);
  // After a pick the list closes but the hits stay, to step through with ◀ ▶.
  const [picked, setPicked] = useState<number | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const hits = findSteps(nodes, query);
  useEffect(() => {
    if (picked !== null && hits[picked]?.path !== selected) {
      setQuery("");
      setPicked(null);
    }
  }, [selected]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Only while this view shows and has the focus: the page may be a whole IDE with its own Ctrl+F.
      const root = input.current?.closest(".decider");
      const at = e.target as Node;
      if (!root?.getClientRects().length || !(root.contains(at) || at.contains(root))) return;
      if ((e.ctrlKey || e.metaKey) && e.key === "f") {
        e.preventDefault();
        input.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const pick = (i: number) => {
    const n = hits[i];
    if (!n) return;
    onPick(n.path);
    setPicked(i);
  };
  const step = (d: number) => pick(((picked ?? 0) + d + hits.length) % hits.length);
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
          setPicked(null);
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") setAt(Math.min(at + 1, Math.min(hits.length, MAX_HITS) - 1));
          else if (e.key === "ArrowUp") setAt(Math.max(at - 1, 0));
          else if (e.key === "Enter") picked === null ? pick(at) : step(e.shiftKey ? -1 : 1);
          else if (e.key === "Escape") {
            setQuery("");
            setPicked(null);
          }
        }}
      />
      {query && picked !== null && hits.length > 1 && (
        <span className="find-nav">
          <button title="Previous hit (Shift+Enter)" onClick={() => step(-1)}>◀</button>
          {picked + 1} of {hits.length}
          <button title="Next hit (Enter)" onClick={() => step(1)}>▶</button>
        </span>
      )}
      {query && picked === null && (
        <ul className="find-hits" role="listbox">
          {hits.slice(0, MAX_HITS).map((n, i) => (
            <li key={n.path} role="option" aria-selected={i === at} className={i === at ? "at" : ""} onMouseDown={() => pick(i)}>
              {tagOf(n.path) && <span className="tag">{tagOf(n.path)}</span>}
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
