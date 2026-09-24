import { formatValue, type ValueChange, type ValueHistory } from "../src/protocol";

interface Props {
  history: ValueHistory;
  /** The focused record's label, or null for the whole batch. */
  who: string | null;
  onSelect: (path: string) => void;
  onGoTo: (change: number) => void;
}

const isOverride = (c: ValueChange) => c.path.startsWith("override@");
const by = (c: ValueChange) => (isOverride(c) ? "you" : c.path.split("/").pop());
const when = (c: ValueChange) => (c.iteration ? ` in iteration ${c.iteration}` : "");

/** Each change to a value so far, oldest first, with the step that made it and a way back to that moment. */
export function ValueTimeline({ history, who, onSelect, onGoTo }: Props) {
  const { name, changes } = history;
  const fmt = (v: unknown) => formatValue(v, name);
  const last = changes[changes.length - 1];
  const step = (c: ValueChange) => (isOverride(c) ? <span>you</span> : <a title={c.path} onClick={() => onSelect(c.path)}>{by(c)}</a>);
  const goBack = (c: ValueChange, text: string) =>
    !isOverride(c) && (
      <button className="link" title={`Re-run to just after ${by(c)}${when(c)} and pause there, with every value as it was then`} onClick={() => onGoTo(c.change)}>
        {text}
      </button>
    );
  return (
    <section className="timeline">
      <h4>How {name} changed{who ? ` for ${who}` : ""}</h4>
      {who && last && (
        <div className="why">
          {name} is <strong>{fmt(last.value)}</strong> because {step(last)} set it{when(last)} (it was {fmt(last.before)}). {goBack(last, "⤺ Go back to that moment")}
        </div>
      )}
      {who && !last && <div className="muted">{history.input ? `${name} is still its input value, ${fmt(history.initial)}.` : `Nothing has set ${name} yet.`}</div>}
      <ol className="history">
        <li className="muted">{history.input ? <>starts as {who ? <span className="mono">{fmt(history.initial)}</span> : "the input column"}</> : "not set at the start"}</li>
        {changes.map((c) => (
          <li key={c.change}>
            {who ? (
              <span className="mono">{fmt(c.before)} → <strong>{fmt(c.value)}</strong></span>
            ) : (
              <span className="mono">{c.rows} record{c.rows === 1 ? "" : "s"}: {c.values!.map(fmt).join(", ")}{c.rows! > c.values!.length ? ", …" : ""}</span>
            )}{" "}
            <span className="muted">by</span> {step(c)}
            <span className="muted">{when(c)}</span> {goBack(c, "go back here")}
          </li>
        ))}
      </ol>
    </section>
  );
}
