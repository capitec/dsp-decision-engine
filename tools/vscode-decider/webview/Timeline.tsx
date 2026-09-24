import { same } from "../src/compare";
import { formatValue, type Lineage, type ValueChange, type ValueHistory } from "../src/protocol";

interface Props {
  history: ValueHistory;
  /** The focused record's label, or null for the whole batch. */
  who: string | null;
  /** The value's breakdown for the focused record: says where a value a step passed through came from. */
  lineage?: Lineage | null;
  onSelect: (path: string) => void;
  onGoTo: (change: number) => void;
}

const byYou = (c: ValueChange) => c.path.startsWith("override@") || c.path.startsWith("force@");
const short = (path: string) => path.split("/").pop();
const when = (c: ValueChange) => (c.iteration ? ` in iteration ${c.iteration}` : "");

/** Each change to a value so far, oldest first, with the step that made it and a way back to that moment. */
export function ValueTimeline({ history, who, lineage, onSelect, onGoTo }: Props) {
  const { name, changes } = history;
  const fmt = (v: unknown) => formatValue(v, name);
  const last = [...changes].reverse().find((c) => !c.kept && !c.pending);
  // A step that returned one of its inputs unchanged (a cap that didn't bind) passed the value on; name its origin.
  const setter = last && lineage ? find(lineage, name, last.path) : undefined;
  const passed = setter?.inputs.find((i) => same(i.value, last!.value) && i.producer);
  const step = (c: ValueChange) =>
    c.path.startsWith("force@") ? (
      <span>you, forcing {short(c.path.slice(6))}</span>
    ) : c.path.startsWith("override@") ? (
      <span>you</span>
    ) : (
      <a title={c.path} onClick={() => onSelect(c.path)}>{short(c.path)}</a>
    );
  const goBack = (c: ValueChange, text: string) =>
    !byYou(c) && (
      <button className="link" title={`Re-run to just after ${short(c.path)}${when(c)} and pause there, with every value as it was then`} onClick={() => onGoTo(c.change)}>
        {text}
      </button>
    );
  return (
    <section className="timeline">
      <h4>How {name} changed{who ? ` for ${who}` : ""}</h4>
      {who && last && (
        <div className="why">
          {name} is <strong>{fmt(last.value)}</strong> because {step(last)} set it{when(last)}
          {last.before !== null && last.before !== undefined ? ` (it was ${fmt(last.before)})` : ""}. {goBack(last, "⤺ Go back to that moment")}
          {passed && (
            <div className="small">
              It passed <span className="mono">{passed.name}</span> through unchanged, so the value comes from{" "}
              <a title={passed.producer!} onClick={() => onSelect(passed.producer!)}>{short(passed.producer!)}</a>.
            </div>
          )}
        </div>
      )}
      {who && !last && <div className="muted">{history.input ? `${name} is still its input value, ${fmt(history.initial)}.` : `Nothing has set ${name} yet.`}</div>}
      <ol className="history">
        {history.input && <li className="muted">starts as {who ? <span className="mono">{fmt(history.initial)}</span> : "the input column"}</li>}
        {changes.map((c) => (
          <li key={`${c.change}-${c.path}-${c.iteration}`} className={c.pending ? "pending" : c.kept ? "kept" : ""} title={c.pending ? "Undone by going back: it runs again when you continue" : undefined}>
            {c.kept ? (
              <span className="mono muted">kept {fmt(c.value)}</span>
            ) : who ? (
              c.before === null || c.before === undefined ? (
                <span className="mono">set to <strong>{fmt(c.value)}</strong></span>
              ) : (
                <span className="mono">{fmt(c.before)} → <strong>{fmt(c.value)}</strong></span>
              )
            ) : (
              <span className="mono">{c.rows} record{c.rows === 1 ? "" : "s"}: {c.values!.map(fmt).join(", ")}{c.rows! > c.values!.length ? ", …" : ""}</span>
            )}{" "}
            <span className="muted">by</span> {step(c)}
            <span className="muted">{when(c)}</span> {c.pending ? <span className="muted small">· will re-run</span> : goBack(c, "go back here")}
          </li>
        ))}
      </ol>
    </section>
  );
}

/** The entry for `name` as `producer` wrote it, anywhere in a breakdown. */
function find(l: Lineage, name: string, producer: string): Lineage | undefined {
  if (l.name === name && l.producer === producer) return l;
  for (const i of l.inputs) {
    const hit = find(i, name, producer);
    if (hit) return hit;
  }
  return undefined;
}
