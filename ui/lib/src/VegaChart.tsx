import { useEffect, useRef } from "react";
import type { Figure } from "./model/experiment";

/** One figure, drawn with `vega-embed`; handles either grammar from the same `spec`. */
export function VegaChart({ figure }: { figure: Figure }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let view: { finalize: () => void } | undefined;
    el.textContent = "";
    import("vega-embed")
      .then(({ default: embed }) => embed(el, figure.spec, { mode: figure.grammar, actions: false }))
      .then((result) => {
        view = result.view;
      })
      .catch((err: Error) => {
        el.innerHTML = `<pre class="chart-error">${err.message}</pre>`;
      });
    return () => view?.finalize();
  }, [figure]);

  return <div className="vega-chart" ref={ref} />;
}
