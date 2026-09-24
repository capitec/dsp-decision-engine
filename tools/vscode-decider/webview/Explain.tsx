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
  return filled.replace(/ [+-] 0%?(?![.\d%])/g, "").replace(/ \* 1(?![.\d%])/g, "").replace(/(^|[(\s])1 \* /g, "$1");
}

/** A formula that only adds and subtracts names, as signed terms: `a + b - c` gives `[["+", "a"], ["+", "b"], ["-", "c"]]`. */
export function sumTerms(formula: string): ["+" | "-", string][] | null {
  if (!/^\s*[A-Za-z_]\w*(\s*[+-]\s*[A-Za-z_]\w*)+\s*$/.test(formula)) return null;
  const terms: ["+" | "-", string][] = [];
  for (const m of formula.matchAll(/([+-]?)\s*([A-Za-z_]\w*)/g)) terms.push([m[1] === "-" ? "-" : "+", m[2]]);
  return terms;
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
        Why <span className="mono">{entry.name}{who ? ` = ${formatValue(entry.value, entry.name)}` : ""}</span>
        {who ? ` for ${who}` : ""}
        {entry.producer && <span className="muted"> · last written by {short(entry.producer)}</span>}
        {role && <span className="muted small"> ({role})</span>}
      </div>
      {who && entry.producer !== null && summaryRows(entry, nodes, values) && (
        <div className="verdict-line">{verdict(summaryRows(entry, nodes, values)!)}</div>
      )}
      {who && entry.producer !== null && summaryRows(entry, nodes, values) && (
        <table className="explain-summary">
          <tbody>
            {summaryRows(entry, nodes, values)!.map((r, k) => (
              <tr key={k} className={r.kind}>
                <td>{r.label}</td>
                <td className="mono num">{r.value}</td>
                <td className="muted small">{r.note ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
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

/** One row of the summary table: a part of the value, a limit it met, or the total. */
interface SummaryRow {
  label: string;
  value: string;
  note?: string;
  kind: "part" | "zero" | "total" | "limit";
}

/** "25.2% = pl_base_rate 25.5% − 0.3% from 2 parts · cap not reached · floor not reached". */
function verdict(rows: SummaryRow[]): string {
  const parts = rows.filter((r) => r.kind === "part");
  const [base, ...rest] = parts;
  const adjust = rest.length ? ` ${rest.map((r) => r.value).join(" ")}` : "";
  const limits = rows.filter((r) => r.kind === "limit").map((r) => `${r.label.split(" ")[0]} ${r.note?.startsWith("not") ? "not reached" : "applied"}`);
  const total = rows[rows.length - 1];
  return `${total.value} = ${base ? `${base.label} ${base.value}` : ""}${adjust}${limits.length ? ` · ${limits.join(" · ")}` : ""}`;
}

/** Down the chain of caps and floors to the step that builds the value, as rows: the parts, the sum, then each limit. */
export function summaryRows(entry: Lineage, nodes: CallNodeJson[], values: Record<string, unknown>): SummaryRow[] | null {
  const limits: SummaryRow[] = [];
  let at: Lineage | undefined = entry;
  for (let depth = 0; at && depth < 6; depth++) {
    const node = nodes.find((n) => n.path === at!.producer);
    if (!node?.formula) break;
    const known: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(node.params)) known[k] = shared(values)[k] ?? v;
    for (const i of at.inputs) known[i.name] = i.value;
    const note = clampNote(node.formula, at.inputs, at.value, known);
    if (note) {
      const [head, tail] = note.includes(", ") ? note.split(/, (.*)/s) : [note, ""];
      limits.unshift({ label: `${head.split(" ")[0]} (${short(node.path)})`, value: head.split(" ").slice(1).join(" "), note: tail || note, kind: "limit" });
      const first = /^(?:min|max)\((\w+),/.exec(node.formula)![1];
      at = at.inputs.find((i) => i.name === first);
      continue;
    }
    const terms = sumTerms(node.formula);
    if (!terms || limits.length === 0) return null;
    const part = (name: string) => at!.inputs.find((i) => i.name === name);
    const rows: SummaryRow[] = [];
    const zeros: string[] = [];
    terms.forEach(([sign, name], k) => {
      const p = part(name);
      if (p?.value === 0) zeros.push(name);
      const from = nodes.find((x) => x.path === p?.producer);
      const tableRows = from?.table ? (Object.keys(from.params).map((q) => shared(values)[q]).find(Array.isArray) as Record<string, unknown>[] | undefined) : undefined;
      const match = from?.table && tableRows && p?.inputs[0] ? matchRow(from.table, tableRows, p.inputs[0].value) : null;
      rows.push({ label: name, value: `${k === 0 ? "" : sign === "-" ? "− " : "+ "}${formatValue(p?.value as never, name)}`, note: match ? `row ${match[0] + 1}: ${match[1]}` : undefined, kind: p?.value === 0 ? "zero" : "part" });
    });
    rows.push({ label: `= ${at.name}`, value: formatValue(at.value, at.name), kind: "total" });
    return [...rows, ...limits, { label: `= ${entry.name}`, value: formatValue(entry.value, entry.name), kind: "total" }];
  }
  return null;
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
  // A sum shows as a waterfall, which already names each part; its inputs' own breakdowns open on request.
  const terms = node?.formula ? sumTerms(node.formula) : null;
  const [partsOpen, setPartsOpen] = useState(false);
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
      {open && who && terms && (
        <table className="waterfall">
          <tbody>
            {terms.map(([sign, n], k) => {
              const part = entry.inputs.find((i) => i.name === n);
              const from = nodes.find((x) => x.path === part?.producer);
              const partRows = from?.table ? (Object.keys(from.params).map((p) => shared(values)[p]).find(Array.isArray) as Record<string, unknown>[] | undefined) : undefined;
              const row = from?.table && partRows && part?.inputs[0] ? matchRow(from.table, partRows, part.inputs[0].value) : null;
              return (
                <tr key={n} className={part?.value === 0 ? "zero" : ""}>
                  <td>
                    <a title={part?.producer ? `Show ${short(part.producer)} in the graph` : n} onClick={() => part?.producer && onSelect(part.producer)}>{n}</a>
                  </td>
                  <td className="mono num">{k === 0 ? "" : sign === "-" ? "− " : "+ "}{formatValue(part?.value as never, n)}</td>
                  <td className="muted small">{row ? `row ${row[0] + 1} of ${short(from!.path)}: ${row[1]}` : ""}</td>
                </tr>
              );
            })}
            <tr className="total">
              <td>= {entry.name}</td>
              <td className="mono num">{formatValue(entry.value, entry.name)}</td>
              <td />
            </tr>
          </tbody>
        </table>
      )}
      {open && node?.formula && who && !terms && (
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
      {open && terms && entry.inputs.length > 0 && !partsOpen && (
        <div className="small">
          <span className="twisty" />
          <a onClick={() => setPartsOpen(true)}>▸ how each part was computed</a>
        </div>
      )}
      {open && entry.inputs.length > 0 && (!terms || partsOpen) && (
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
