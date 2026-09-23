import { useEffect, useRef, useState } from "react";
import { formatValue, isRateName, type CallNodeJson, type Lineage } from "../src/protocol";

interface Props {
  entry: Lineage;
  who: string | null;
  role: string;
  nodes: CallNodeJson[];
  /** The flow's PARAMS document: shared values and lookup tables' rows. */
  values: Record<string, unknown>;
  onPick: (name?: string) => void;
  onSelect: (path: string) => void;
}

// Levels opened at first: enough to show a rate's build-up without burying it.
const OPEN_DEPTH = 3;

const short = (path: string) => path.slice(path.lastIndexOf("/") + 1);
const shared = (values: Record<string, unknown>) => (values.shared ?? {}) as Record<string, unknown>;

/** `formula` with each name replaced by its value: "min(0.252, 0.0775 * 1 + 0.21)". */
export function substitute(formula: string, known: Record<string, unknown>): string {
  const filled = formula.replace(/\b[A-Za-z_]\w*\b/g, (name) => (name in known ? formatValue(known[name] as never, name) : name));
  // A sum of many terms reads better without the ones that are zero.
  return filled.replace(/ [+-] 0%?(?![.\d%])/g, "");
}

/** `+ - * /` and parentheses over numbers and known names; null when it can't (a call, a comparison…). */
export function evaluate(expr: string, known: Record<string, unknown>): number | null {
  const tokens = expr.match(/\d+\.?\d*(?:e[+-]?\d+)?|[A-Za-z_]\w*|[-+*/()]/g) ?? [];
  if (tokens.join("") !== expr.replace(/\s+/g, "")) return null;
  let i = 0;
  const atom = (): number | null => {
    const t = tokens[i++];
    if (t === "(") {
      const v = sum();
      return tokens[i++] === ")" ? v : null;
    }
    if (t === "-") {
      const v = atom();
      return v === null ? null : -v;
    }
    if (t !== undefined && /^\d/.test(t)) return Number(t);
    return t !== undefined && typeof known[t] === "number" ? (known[t] as number) : null;
  };
  const product = (): number | null => {
    let v = atom();
    while (v !== null && (tokens[i] === "*" || tokens[i] === "/")) {
      const op = tokens[i++];
      const r = atom();
      v = r === null ? null : op === "*" ? v * r : v / r;
    }
    return v;
  };
  const sum = (): number | null => {
    let v = product();
    while (v !== null && (tokens[i] === "+" || tokens[i] === "-")) {
      const op = tokens[i++];
      const r = product();
      v = r === null ? null : op === "+" ? v + r : v - r;
    }
    return v;
  };
  const v = sum();
  return i === tokens.length ? v : null;
}

/** For `min(x, limit)` / `max(x, limit)`: the limit and whether it changed `x`, e.g. "cap 28.75%, not reached (3.55 pp below)". */
export function clampNote(formula: string, inputs: { name: string; value: unknown }[], result: unknown, known: Record<string, unknown> = {}): string | null {
  const m = /^(min|max)\((\w+),\s*(.*)\)$/.exec(formula);
  const first = m && inputs.find((i) => i.name === m[2]);
  if (!m || !first || typeof result !== "number" || typeof first.value !== "number") return null;
  const kind = m[1] === "min" ? "cap" : "floor";
  const limit = evaluate(m[3], known);
  const shown = (v: number) => formatValue(v, first.name);
  const gap = (a: number, b: number) => (isRateName(first.name) ? `${Number((Math.abs(a - b) * 100).toFixed(2))} pp` : formatValue(Math.abs(a - b)));
  if (Math.abs(result - first.value) < 1e-12) {
    return limit === null
      ? `the ${kind} did not apply: ${first.name} passed through`
      : `${kind} ${shown(limit)}, not reached (${gap(limit, first.value)} ${kind === "cap" ? "below" : "above"})`;
  }
  return `the ${kind} applied: ${shown(first.value)} → ${shown(result)}`;
}

/** The row of a lookup table a value falls in, and why: `[2, "49 ≤ requested_term 60 < 85"]`. */
export function matchRow(expr: Record<string, unknown>, rows: Record<string, unknown>[], x: unknown): [number, string] | null {
  if (expr.type === "between") {
    const lo = expr.lower_bound_column as string | undefined;
    const hi = expr.upper_bound_column as string | undefined;
    const upperInclusive = expr.mode === "upper_inclusive";
    const i = rows.findIndex((r) => {
      const a = lo ? (r[lo] as number | null) : null;
      const b = hi ? (r[hi] as number | null) : null;
      const n = x as number;
      return (a === null || (upperInclusive ? n > a : n >= a)) && (b === null || (upperInclusive ? n <= b : n < b));
    });
    if (i < 0) return null;
    const r = rows[i];
    const left = lo && r[lo] !== null ? `${formatValue(r[lo] as never)} ${upperInclusive ? "<" : "≤"} ` : "";
    const right = hi && r[hi] !== null ? ` ${upperInclusive ? "≤" : "<"} ${formatValue(r[hi] as never)}` : "";
    return [i, `${left}${expr.variable} ${formatValue(x as never)}${right}`];
  }
  if (expr.type === "eq") {
    const col = expr.value_column as string;
    const i = rows.findIndex((r) => r[col] === x);
    return i < 0 ? null : [i, `${expr.variable} = ${formatValue(x as never)}`];
  }
  return null;
}

/** How a value was computed for the focused record: each step's formula with the values it had, level by level. */
export function Explain({ entry, who, role, nodes, values, onPick, onSelect }: Props) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => box.current?.scrollIntoView({ block: "start", behavior: "smooth" }), [entry.name, entry.producer]);
  return (
    <div className="how" ref={box}>
      <div className="how-title">
        {who ? `For ${who}, ` : ""}
        <span className="mono">{entry.name}{who ? ` = ${formatValue(entry.value, entry.name)}` : ""}</span>
        {role && <span className="muted"> ({role})</span>}
      </div>
      {entry.producer !== null && summary(entry, nodes, values) && (
        <div className="explain-summary">
          {summary(entry, nodes, values)!.split(" · ").map((part) => (
            <div key={part}>{part}</div>
          ))}
        </div>
      )}
      {entry.producer === null ? (
        <div>It is an input: it arrives with the data.</div>
      ) : (
        <ul className="explain">
          <Level entry={entry} depth={0} nodes={nodes} values={values} who={who} onPick={onPick} onSelect={onSelect} />
        </ul>
      )}
      <div className="muted small">Click a value to open its own inputs; a step name selects it in the graph.</div>
    </div>
  );
}

/** Down the chain of caps and floors to the step that builds the value: "floor did not apply · cap did not apply · pl_raw_rate = 25.5% − 0.1% − 0.2%". */
function summary(entry: Lineage, nodes: CallNodeJson[], values: Record<string, unknown>): string | null {
  const parts: string[] = [];
  let at: Lineage | undefined = entry;
  for (let depth = 0; at && depth < 6; depth++) {
    const node = nodes.find((n) => n.path === at!.producer);
    if (!node?.formula) break;
    const known: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(node.params)) known[k] = shared(values)[k] ?? v;
    for (const i of at.inputs) known[i.name] = i.value;
    const note = clampNote(node.formula, at.inputs, at.value, known);
    if (note) {
      parts.push(note.replace(/: .*$/, ""));
      const first = /^(?:min|max)\((\w+),/.exec(node.formula)![1];
      at = at.inputs.find((i) => i.name === first);
      continue;
    }
    parts.push(`${at.name} = ${substitute(node.formula, known)}`);
    break;
  }
  return parts.length > 1 ? parts.join(" · ") : null;
}

function Level({ entry, depth, nodes, values, who, onPick, onSelect }: { entry: Lineage; depth: number; nodes: CallNodeJson[]; values: Record<string, unknown>; who: string | null; onPick: (n?: string) => void; onSelect: (p: string) => void }) {
  const [open, setOpen] = useState(depth < OPEN_DEPTH);
  const [showZeros, setShowZeros] = useState(false);
  // A term that is 0 for this record adds nothing to the answer; folded unless asked for, or when it's the only one.
  const zero = (i: Lineage) => i.value === 0 && entry.inputs.filter((x) => x.value === 0).length > 1;
  const node = nodes.find((n) => n.path === entry.producer);
  const known: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(node?.params ?? {})) known[k] = shared(values)[k] ?? v;
  for (const i of entry.inputs) known[i.name] = i.value;
  const rows = node?.table ? (Object.keys(node.params).map((k) => shared(values)[k]).find(Array.isArray) as Record<string, unknown>[] | undefined) : undefined;
  const match = node?.table && rows && entry.inputs[0] ? matchRow(node.table, rows, entry.inputs[0].value) : null;
  const expandable = entry.producer !== null && (entry.inputs.length > 0 || !!node?.formula);
  return (
    <li>
      {expandable ? (
        <button className="link twisty" aria-expanded={open} onClick={() => setOpen(!open)}>{open ? "▾" : "▸"}</button>
      ) : (
        <span className="twisty" />
      )}
      <a className="mono" title={expandable ? `Show how ${entry.name} was computed` : entry.name} onClick={() => expandable && setOpen(!open)}>{entry.name}</a>
      {who && <strong className="mono"> = {formatValue(entry.value, entry.name)}</strong>}
      <span className="muted">
        {" "}
        {entry.producer === null ? "input" : <>from <a onClick={() => onSelect(entry.producer!)}>{short(entry.producer)}</a></>}
        {entry.via === "merge" && " (the branch arm this record took)"}
        {entry.via === "carry" && " (the loop's last iteration)"}
      </span>
      {open && node?.formula && who && (
        <div className="formula mono" title={node.formula}>
          = {substitute(node.formula, known)}
          {clampNote(node.formula, entry.inputs, entry.value, known) && <div className="clamp">{clampNote(node.formula, entry.inputs, entry.value, known)}</div>}
        </div>
      )}
      {open && match && (
        <div className="formula">
          row {match[0] + 1} of <span className="mono">{short(node!.path)}</span>: {match[1]}
        </div>
      )}
      {open && entry.inputs.length > 0 && (
        <ul>
          {entry.inputs.filter((i) => !zero(i) || showZeros).map((i, k) => (
            <Level key={k} entry={i} depth={depth + 1} nodes={nodes} values={values} who={who} onPick={onPick} onSelect={onSelect} />
          ))}
          {!showZeros && entry.inputs.filter(zero).length > 1 && (
            <li className="muted small">
              <span className="twisty" />
              <a onClick={() => setShowZeros(true)}>+ {entry.inputs.filter(zero).length} parts at 0: {entry.inputs.filter(zero).map((i) => i.name).join(", ")}</a>
            </li>
          )}
        </ul>
      )}
    </li>
  );
}
