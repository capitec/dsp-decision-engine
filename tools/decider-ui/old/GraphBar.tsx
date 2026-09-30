/** The graph's view options and what its marks mean. */
export function ViewMenu({ showData, onShowData, details, onDetails }: { showData: boolean; onShowData: (v: boolean) => void; details: boolean; onDetails: (v: boolean) => void }) {
  return (
    <details className="legend-pop">
      <summary>View</summary>
      <div className="legend-body">
        <label><input type="checkbox" checked={showData} onChange={(e) => onShowData(e.target.checked)} /> show all data links</label>
        <label><input type="checkbox" checked={details} onChange={(e) => onDetails(e.target.checked)} /> step details</label>
        <span><span className="line solid" /> runs next</span>
        <span><span className="line dotted" /> passes data (for the selected step)</span>
        <span><span className="swatch paused" /> paused here</span>
        <span><span className="swatch lineage" /> inputs of the picked value</span>
        <span>✓ has run</span>
        <span>◇ decision tree</span>
        <span>▦ lookup table</span>
        <span><span className="swatch changed" /> changed in the last comparison</span>
        <span><span className="swatch added" /> added in the last comparison</span>
        <span>⊞ data frame step</span>
      </div>
    </details>
  );
}

/** Colour the graph by the last comparison, and step through its changed steps: the edited ones, then the knock-on ones. */
export function ChangedNav({ steps, edited, selected, showDiff, onShowDiff, onSelect }: { steps: string[]; edited: number; selected?: string; showDiff: boolean; onShowDiff: (v: boolean) => void; onSelect: (path: string) => void }) {
  const at = steps.indexOf(selected ?? "");
  const go = (dir: 1 | -1) => onSelect(steps[(at + dir + steps.length) % steps.length]);
  return (
    <span className="legend">
      <label title="Colour the graph by the last comparison: orange changed, green added"><input type="checkbox" checked={showDiff} onChange={(e) => onShowDiff(e.target.checked)} /> changes</label>
      {showDiff && steps.length > 0 && (
        <>
          <button title="Previous changed step" onClick={() => go(-1)}>◀</button>
          <span title={steps[at]}>
            {at >= 0
              ? at < edited
                ? `edit ${at + 1} of ${edited}: ${steps[at].split("/").pop()}`
                : `result ${at - edited + 1} of ${steps.length - edited}: ${steps[at].split("/").pop()}`
              : `changed steps: ${edited ? `${edited} edited, ` : ""}${steps.length - edited} as a result`}
          </span>
          <button title="Next changed step" onClick={() => go(1)}>▶</button>
        </>
      )}
    </span>
  );
}

export function Key() {
  return (
    <details className="legend">
      <summary>Key</summary>
      <div>
        <span className="swatch paused" /> paused here · <span className="swatch selected" /> selected · ✓ has run
        <br />
        <span className="swatch lineage" /> where the value you picked came from · dotted arcs: that value's data flow
        <br />
        <span className="swatch changed" /> changed in the comparison · faded: unchanged (untick "changes" to hide)
      </div>
    </details>
  );
}
