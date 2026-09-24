import { formatValue, type Checkpoint, type ValueChange, type ValueHistory } from "../src/protocol";

interface Props {
  history: ValueHistory;
  /** The focused record's label, or null for the whole batch. */
  who: string | null;
  /** Where the debug run is paused, to mark the change it's paused just after. */
  current?: Checkpoint | null;
  onSelect: (path: string) => void;
  /** A change's index, or the path of a step to go back to just after. */
  onGoTo: (change: number | string) => void;
}

const byYou = (c: ValueChange) => c.path.startsWith("override@") || c.path.startsWith("force@");
const short = (path: string) => path.split("/").pop();
const when = (c: ValueChange) => (c.iteration ? ` in iteration ${c.iteration}` : "");

/** Each change to a value so far, oldest first, with the step that made it and a way back to that moment. */
export function ValueTimeline({ history, who, current, onSelect, onGoTo }: Props) {
  const { name, changes } = history;
  const fmt = (v: unknown) => formatValue(v, name);
  const done = changes.filter((c) => !c.pending);
  const last = [...done].reverse().find((c) => !c.kept);
  const next = changes.find((c) => c.pending && !c.kept);
  const keptBy = last ? done.filter((c) => c.kept && done.indexOf(c) > done.indexOf(last)).map((c) => short(c.path)) : [];
  const isHere = (path: string, iteration?: number | null) =>
    !!current && current.when === "after" && current.path === path && (current.iteration ?? null) === (iteration ?? null);
  const link = (path: string) => (
    <a title={path} onClick={() => onSelect(path)}>{short(path)}</a>
  );
  const step = (c: ValueChange) =>
    c.path.startsWith("force@") ? <span>you, forcing {short(c.path.slice(6))}</span> : c.path.startsWith("override@") ? <span>you</span> : link(c.path);
  const back = (target: number | string, path: string, iteration: number | null | undefined, text: string) =>
    isHere(path, iteration) ? (
      <span className="here">◀ you are here</span>
    ) : (
      <button className="link" title={`Re-run to just after ${short(path)}${iteration ? ` in iteration ${iteration}` : ""} and pause there, with every value as it was then`} onClick={() => onGoTo(target)}>
        {text}
      </button>
    );
  const origin = (c: ValueChange) =>
    c.viaPath ? (
      <>
        {link(c.viaPath)}, which computed it as <span className="mono">{c.via}</span>
      </>
    ) : (
      <>
        the data, where it arrived as <span className="mono">{c.via}</span>
      </>
    );

  return (
    <section className="timeline">
      <h4>How {name} changed{who ? ` for ${who}` : ""}</h4>
      {who && last && (
        <div className="why">
          {last.via ? (
            <>
              {name} is <strong>{fmt(last.value)}</strong>. The number comes from {origin(last)}; {step(last)} copied it into {name} unchanged
              {keptBy.length ? ` and ${keptBy.join(", ")} kept it` : ""}.{" "}
              {last.viaPath ? back(last.viaPath, last.viaPath, null, "⤺ Go back to where it was computed") : back(last.change, last.path, last.iteration, "⤺ Go back to that moment")}
            </>
          ) : (
            <>
              {name} is <strong>{fmt(last.value)}</strong> because {step(last)} set it{when(last)}
              {last.before !== null && last.before !== undefined ? ` (it was ${fmt(last.before)})` : ""}
              {keptBy.length ? `; ${keptBy.join(", ")} kept it` : ""}. {!byYou(last) && back(last.change, last.path, last.iteration, "⤺ Go back to that moment")}
            </>
          )}
        </div>
      )}
      {who && !last && (
        <div className="why">
          {next?.via ? (
            <>
              Nothing has set {name} yet. The number will come from {origin(next)} ({fmt(next.value)}); {step(next)} copies it into {name} when the run continues.
            </>
          ) : history.input ? (
            `${name} is still its input value, ${fmt(history.initial)}.`
          ) : (
            `Nothing has set ${name} yet.`
          )}
        </div>
      )}
      <ol className="history">
        {history.input && <li className="muted">starts as {who ? <span className="mono">{fmt(history.initial)}</span> : "the input column"}</li>}
        {changes.flatMap((c) => [
          who && c.via && c.viaPath && !c.kept && (
            <Row key={`o-${c.change}-${c.path}`} pending={false} kept={false}>
              {link(c.viaPath)} <span className="muted">computed</span> <span className="mono">{c.via} = <strong>{fmt(c.value)}</strong></span>{" "}
              {back(c.viaPath, c.viaPath, null, "go back here")}
            </Row>
          ),
          <Row key={`${c.change}-${c.path}-${c.iteration}-${c.pending ? "p" : ""}`} pending={!!c.pending} kept={!!c.kept}>
            {step(c)}
            <span className="muted">{when(c)}</span>{" "}
            {c.kept ? (
              <span className="muted">kept it at <span className="mono">{fmt(c.value)}</span></span>
            ) : who ? (
              c.via ? (
                <span className="muted">
                  copied <span className="mono">{c.via}</span> into <span className="mono">{name}</span> unchanged
                </span>
              ) : c.before === null || c.before === undefined ? (
                <span className="mono">set it to <strong>{fmt(c.value)}</strong></span>
              ) : (
                <span className="mono">{fmt(c.before)} → <strong>{fmt(c.value)}</strong></span>
              )
            ) : (
              <span className="mono">{c.rows} record{c.rows === 1 ? "" : "s"}: {c.values!.map(fmt).join(", ")}{c.rows! > c.values!.length ? ", …" : ""}</span>
            )}{" "}
            {c.pending ? <span className="muted small">· will run again when you continue</span> : !byYou(c) && !c.kept && !c.via && back(c.change, c.path, c.iteration, "go back here")}
          </Row>,
        ])}
      </ol>
    </section>
  );
}

function Row({ pending, kept, children }: { pending: boolean; kept: boolean; children: React.ReactNode }) {
  return (
    <li className={pending ? "pending" : kept ? "kept" : ""} title={pending ? "Undone by going back: it runs again when you continue" : undefined}>
      {children}
    </li>
  );
}
