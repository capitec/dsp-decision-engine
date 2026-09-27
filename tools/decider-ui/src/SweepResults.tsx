import { same } from "./model/compare";
import { formatValue, isPercent, recordLabel } from "./model/protocol";
import type { Sweep } from "./model/sweep";

const categorical = (v: unknown[] | undefined) => !!v?.length && v.every((x) => typeof x === "string" || typeof x === "boolean");

/** "pl_product_cap cap" for a step's param, "repo_rate" for a shared one, the field for an input. */
function knobShort(name: string): string {
  const [path, param] = name.split(" · ");
  if (param === undefined || path === "shared") return param ?? name;
  const step = path.split("/").pop()!;
  return step.includes(param) ? step : `${step} ${param}`;
}

export function SweepResults({ sweep, row, rows, onRow, onOpen, open }: { sweep: Sweep; row: number; rows: number; onRow: (r: number) => void; onOpen: (i: number) => void; open: number | null }) {
  // The outcomes that moved most come first, so the columns that matter survive a narrow panel.
  const moved = (c: string) => sweep.comparisons.reduce((t, cmp) => t + (cmp.output.find((o) => o.name === c)?.changedRows.length ?? 0), 0);
  const changedCols = [...sweep.changedColumns].sort((a, b) => moved(b) - moved(a));
  // Outcomes with a few values (approve / refer / decline) always show in the summary, changed or not.
  const outcomes = Object.entries(sweep.base ?? {})
    .filter(([c, v]) => !changedCols.includes(c) && !sweep.comparisons[0]?.inputColumns.includes(c) && v.every((x) => typeof x === "string") && new Set(v).size <= 4)
    .map(([c]) => c)
    .slice(0, 2);
  const knobCols = sweep.knobs.length ? sweep.knobs : [{ name: "scenario", values: sweep.labels }];
  const summary = row === -1;
  const each = row === -2;
  const fewValues = (c: string) => new Set(sweep.base?.[c] ?? []).size <= 4;
  const cols = summary
    ? [...outcomes, ...changedCols.filter((c) => categorical(sweep.base?.[c]) && fewValues(c)), ...changedCols.filter((c) => !categorical(sweep.base?.[c]))]
    : changedCols;
  // Side by side, the records some scenario changed come first; a few fit.
  const hit = (r: number) => sweep.comparisons.some((cmp) => cmp.output.some((o) => o.changedRows.includes(r)));
  const everyRow = Array.from({ length: rows }, (_, i) => i);
  const shown = each ? [...everyRow.filter(hit), ...everyRow.filter((r) => !hit(r))].slice(0, 3) : [row];
  const recordsOf = (list: number[]) => list.map((r) => recordLabel(r, sweep.key)).join(", ");
  const baseKnob = (name: string) => {
    const values = sweep.knobBase[name] ?? [];
    if (!summary && !each) return formatValue(values[row], knobShort(name));
    return values.every((v) => same(v, values[0])) ? formatValue(values[0], knobShort(name)) : values.map((x) => formatValue(x, knobShort(name))).join(" / ");
  };
  // Counts for an outcome ("approve 12 · decline 25"), the average for a number, each against the original run.
  const overall = (values: unknown[] | undefined, base?: unknown[], name?: string) => {
    const v = values ?? [];
    if (categorical(v)) {
      const count = (xs: unknown[] | undefined, k: string) => (xs ?? []).filter((x) => x === k).length;
      return [...new Set([...v, ...(base ?? [])] as string[])]
        .sort()
        .map((k) => {
          const d = base ? count(v, k) - count(base, k) : 0;
          return `${k} ${count(v, k)}${d ? ` (${d > 0 ? "+" : ""}${d})` : ""}`;
        })
        .join(" · ");
    }
    const nums = v.filter((x): x is number => typeof x === "number");
    if (!nums.length) return "varies";
    const avg = (xs: number[]) => xs.reduce((t, x) => t + x, 0) / xs.length;
    const baseNums = (base ?? []).filter((x): x is number => typeof x === "number");
    const d = baseNums.length ? avg(nums) - avg(baseNums) : 0;
    return `avg ${formatValue(avg(nums), name)}${Math.abs(d) > 1e-12 ? ` (${d > 0 ? "+" : "−"}${formatValue(Math.abs(d), name)})` : ""}`;
  };
  // A number's change is averaged over the records it moved, so a 3-record effect isn't diluted by 37 that didn't move.
  const avgDelta = (c: string, i: number, changedRows: number[]) => {
    const ds = changedRows.map((r) => (sweep.outputs[i]?.[c]?.[r] as number) - (sweep.base?.[c]?.[r] as number)).filter((d) => !Number.isNaN(d));
    return ds.length ? ds.reduce((t, x) => t + x, 0) / ds.length : 0;
  };
  const hasDecision = !!sweep.base?.decision;
  // A scenario that moved no offered applicant anywhere: one line for the cell, not one per metric.
  const changedIn = (i: number) => [...new Set(sweep.comparisons[i].output.flatMap((o) => o.changedRows))];
  const noOfferIn = (i: number) => hasDecision && changedIn(i).length > 0 && offeredRows(i, changedIn(i)).length === 0;
  const offeredRows = (i: number, rows: number[]) =>
    hasDecision ? rows.filter((r) => sweep.outputs[i]?.decision?.[r] !== "decline" || sweep.base?.decision?.[r] !== "decline") : rows;
  const delta = (c: string, i: number, changedRows: number[]) => {
    const d = avgDelta(c, i, changedRows);
    if (!d) return "";
    const size = isPercent(c, d) ? `${Number((Math.abs(d) * 100).toFixed(2))} pp` : formatValue(Math.abs(d), c);
    return `${d > 0 ? "▲ +" : "▼ −"}${size}`;
  };
  // A scenario whose knobs equal the original run's: it is the setting in force now.
  const isCurrent = (i: number) => knobCols.every((k) => k.name === "scenario" || (sweep.knobBase[k.name] ?? []).every((v) => same(v, k.values[i])));
  const summaryLine = (i: number, c: string) => {
    const diff = sweep.comparisons[i].output.find((o) => o.name === c);
    const numeric = !categorical(sweep.base?.[c]);
    const cls = !diff ? "unchanged" : numeric ? (avgDelta(c, i, diff.changedRows) < 0 ? "changed down" : "changed up") : "changed";
    return (
      <div key={c} className={cls} title={diff ? `changed for ${recordsOf(diff.changedRows)}` : "same as the original run for every record"}>
        <span className="muted">{c}</span>{" "}
        <span className="mono">{!diff ? (outcomes.includes(c) ? overall(sweep.outputs[i]?.[c]) : "no change") : cellText(i, c)}</span>
      </div>
    );
  };
  const recordCell = (i: number | null, c: string, r: number) => {
    const before = sweep.base?.[c]?.[r];
    if (i === null) return <td key={`${c}-${r}`} className="mono">{formatValue(before)}</td>;
    const after = sweep.outputs[i]?.[c]?.[r];
    const changed = !same(before, after);
    return (
      <td key={`${c}-${r}`} className={`mono ${changed ? "changed" : "unchanged"}`} title={changed ? `original run: ${formatValue(before)}` : "same as the original run"}>
        {changed ? <><s className="before">{formatValue(before)}</s> {formatValue(after)}</> : `= ${formatValue(after)}`}
      </td>
    );
  };
  // With exactly two knobs, a grid: one knob down, the other across, one result in each cell.
  const grid = summary && knobCols.length === 2 && knobCols.every((k) => k.name !== "scenario");
  // Offers first: they are what a sweep of rates and caps is usually about.
  const numericCols = cols.filter((c) => !categorical(sweep.base?.[c]));
  const shownMetric = numericCols.find((c) => /offer/.test(c)) ?? numericCols[0] ?? cols[0];
  const uniq = (vs: unknown[]) => vs.filter((v, i) => vs.findIndex((w) => same(v, w)) === i);
  const cellText = (i: number, c: string) => {
    const diff = sweep.comparisons[i].output.find((o) => o.name === c);
    if (!diff) return "no change";
    if (categorical(sweep.base?.[c])) return overall(sweep.outputs[i]?.[c], sweep.base?.[c]);
    const offered = offeredRows(i, diff.changedRows);
    const declinedOnly = diff.changedRows.length - offered.length;
    if (!offered.length) return `no offer changed (${declinedOnly} declined only)`;
    return `${delta(c, i, offered)} on ${offered.length} applicant${offered.length === 1 ? "" : "s"}`;
  };
  // A knob whose values never change a result, whatever the other knobs are set to.
  const idleKnobs = knobCols
    .filter((k) => k.name !== "scenario")
    .filter((k) => {
      const groups = new Map<string, number[]>();
      sweep.labels.forEach((_, i) => {
        const key = JSON.stringify(knobCols.filter((o) => o !== k).map((o) => o.values[i]));
        groups.set(key, [...(groups.get(key) ?? []), i]);
      });
      // Same offers and approvals whichever value it takes (declined applicants' internals aside).
      const offerView = (i: number) => JSON.stringify(cols.map((c) => cellText(i, c).replace(/ · .*$/, "")));
      return [...groups.values()].every((is) => is.every((i) => offerView(i) === offerView(is[0])));
    });
  // Cells are tinted by which way offers moved, so the grid can be read at a glance.
  const tintBy = numericCols.find((c) => /offer_amount/.test(c)) ?? numericCols.find((c) => /offer/.test(c));
  const tint = (i: number) => {
    const d = tintBy ? sweep.comparisons[i].output.find((o) => o.name === tintBy) : undefined;
    const rows = d ? offeredRows(i, d.changedRows) : [];
    return rows.length ? (avgDelta(tintBy!, i, rows) > 0 ? "tint-up" : "tint-down") : "";
  };
  const changedRecords = (i: number) => (changedIn(i).length ? `\nchanged: ${recordsOf(changedIn(i))}` : "");
  const example = grid ? sweep.labels.map((_, i) => cellText(i, shownMetric)).find((t) => t !== "no change" && !t.startsWith("no offer")) : undefined;
  // A knob that changes nothing needs one column, not one per value.
  const idle0 = grid && idleKnobs.includes(knobCols[0]);
  const idle1 = grid && idleKnobs.includes(knobCols[1]);
  const rowValues = grid ? (idle0 ? uniq(knobCols[0].values).slice(0, 1) : uniq(knobCols[0].values)) : [];
  const colValues = grid ? (idle1 ? uniq(knobCols[1].values).slice(0, 1) : uniq(knobCols[1].values)) : [];
  const anyOf = (k: (typeof knobCols)[number]) => `any of ${uniq(k.values).map((v) => formatValue(v, knobShort(k.name))).join(" / ")}`;
  const matrix = grid && (
    <div className="sweep-scroll">
      {outcomes.map((c) => (
        <div key={c} className="note">
          <strong>{c === "decision" ? "Approvals" : c} unchanged in all {sweep.labels.length} scenarios:</strong> {overall(sweep.base?.[c])}.
        </div>
      ))}
      {idleKnobs.map((k) => (
        <div key={k.name} className="note">
          <strong>{knobShort(k.name)} ({uniq(k.values).map((v) => formatValue(v, knobShort(k.name))).join(" / ")}) made no difference</strong> to offers or approvals in any scenario: no applicant's decision or offer turns on it at these values, so the grid shows it as one column.
        </div>
      ))}
      <div className="muted small">
        Original run: {cols.filter((c) => !outcomes.includes(c) && !categorical(sweep.base?.[c])).map((c) => `${c} ${overall(sweep.base?.[c], undefined, c)}`).join(" · ")}
        {tintBy && <> · shaded by {tintBy}: <span className="tint-up-key">up</span> <span className="tint-down-key">down</span>; hover a cell for the applicants</>}
      </div>
      <table className="sweep grid">
        <thead>
          <tr>
            <th className="axes" title={knobCols[0].name}>{knobShort(knobCols[0].name).split(" ")[0]} ↓</th>
            <th className="axes" title={knobCols[1].name} colSpan={colValues.length}>{knobShort(knobCols[1].name).split(" ")[0]} →</th>
          </tr>
          <tr>
            <th />
            {colValues.map((v, j) => (
              <th key={j} className="mono">
                {idle1 ? anyOf(knobCols[1]) : formatValue(v, knobShort(knobCols[1].name))}
                {(sweep.knobBase[knobCols[1].name] ?? []).some((b) => same(b, v)) && <div className="muted small">current</div>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rowValues.map((a, r) => (
            <tr key={r}>
              <th className="mono">
                {idle0 ? anyOf(knobCols[0]) : formatValue(a, knobShort(knobCols[0].name))}
                {(sweep.knobBase[knobCols[0].name] ?? []).some((b) => same(b, a)) && <div className="muted small">current</div>}
              </th>
              {colValues.map((b, j) => {
                const i = sweep.labels.findIndex((_, k) => same(knobCols[0].values[k], a) && same(knobCols[1].values[k], b));
                return i < 0 ? (
                  <td key={j} />
                ) : (
                  <td key={j} className={`clickable ${open === i ? "open" : ""} ${cols.every((c) => cellText(i, c) === "no change") ? "unchanged" : "changed"} ${tint(i)}`} title={`See what changed, step by step${changedRecords(i)}`} onClick={() => onOpen(i)}>
                    {isCurrent(i) && <div className="current-tag">the current setting</div>}
                    {cols.every((c) => cellText(i, c) === "no change") ? (
                      <div className="muted">no change</div>
                    ) : noOfferIn(i) ? (
                      <div className="muted">no offer changed · internal values moved for {changedIn(i).length} declined applicant{changedIn(i).length === 1 ? "" : "s"}</div>
                    ) : (
                      <>
                        {cols.filter((c) => !outcomes.includes(c)).map((c) => (
                          <div key={c} className={cellText(i, c) === "no change" ? "muted" : ""}>
                            <span className="muted small">{c}</span> <span className="mono">{cellText(i, c)}</span>
                          </div>
                        ))}
                        {changedIn(i).length > offeredRows(i, changedIn(i)).length && (
                          <div className="muted small">and internal values of {changedIn(i).length - offeredRows(i, changedIn(i)).length} declined applicants</div>
                        )}
                      </>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
  return (
    <section>
      <div className="summary">
        <span>Show</span>
        <select aria-label="scenario record" value={row} onChange={(e) => onRow(Number(e.target.value))}>
          <option value={-2}>{rows > 3 ? "the 3 records that changed most, side by side" : "each record side by side"}</option>
          <option value={-1}>a summary of all {rows} records</option>
          {Array.from({ length: rows }, (_, i) => (
            <option key={i} value={i}>the values for {recordLabel(i, sweep.key)}</option>
          ))}
        </select>
      </div>
      <p className="hint">
        {grid ? "One cell per combination; click one to see what changed, step by step." : "One row per scenario; click one to see what changed, step by step."}
        {grid && example && ` “${example}” is the average change for those applicants.`}{summary && !grid ? " “−R 5,303.95 (3 rec.)” means 3 records changed, by −R 5,303.95 on average." : ""}
      </p>
      {cols.length === 0 ? (
        <div className="muted">No scenario changes any result.</div>
      ) : grid ? (
        matrix
      ) : (
        <div className="sweep-scroll">
        <table className="sweep">
          <thead>
            {each && (
              <tr>
                <th colSpan={knobCols.length} />
                {shown.map((r) => (
                  <th key={r} colSpan={cols.length} className="group-head">{recordLabel(r, sweep.key)}</th>
                ))}
              </tr>
            )}
            <tr>
              {knobCols.map((k) => (
                <th key={k.name} className="knob-col" title={k.name}>{knobShort(k.name)}</th>
              ))}
              {summary ? <th>results</th> : shown.flatMap((r) => cols.map((c) => <th key={`${c}-${r}`}>{c}</th>))}
            </tr>
          </thead>
          <tbody>
            <tr className="original">
              <td className="knob-col" colSpan={knobCols.length} title={knobCols.map((k) => `${k.name} = ${baseKnob(k.name)}`).join("\n")}>
                original run
                <div className="muted small">{knobCols.filter((k) => k.name !== "scenario").map((k) => baseKnob(k.name)).join(" · ")}</div>
              </td>
              {summary ? (
                <td className="stack">
                  {cols.map((c) => (
                    <div key={c}><span className="muted">{c}</span> <span className="mono">{overall(sweep.base?.[c], undefined, c)}</span></div>
                  ))}
                </td>
              ) : (
                shown.flatMap((r) => cols.map((c) => recordCell(null, c, r)))
              )}
            </tr>
            {sweep.labels.map((label, i) => (
              <tr key={i} className={`clickable ${open === i ? "open" : ""}`} title={`${label}: see what changed, step by step`} onClick={() => onOpen(i)}>
                {knobCols.map((k) => (
                  <td key={k.name} className="mono knob-col">{formatValue(k.values[i], knobShort(k.name))}</td>
                ))}
                {summary ? (
                  <td className="stack">
                    {isCurrent(i) && <div className="current-tag">the current setting</div>}
                    {cols.map((c) => summaryLine(i, c))}
                  </td>
                ) : (
                  shown.flatMap((r) => cols.map((c) => recordCell(i, c, r)))
                )}
                {sweep.errors[i] && <td className="error small">{sweep.errors[i]}</td>}
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}
    </section>
  );
}
