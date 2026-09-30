import { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { App, type EditorMessage } from "../src/App";
import type { FromUI, ToUI } from "../src/model/protocol";
import "../src/style.css";
import { largeFlow, pausedRun, smallFlow, experimentResult } from "./mock";
import "./theme.css";

// A mock host: delivers one describe payload, can pause the run, and prints the messages the UI
// would send the extension (opening source, stepping) instead of acting on them.
const can = new Set<EditorMessage>(["reveal", "run", "runTo", "step", "debugStep"]);

const IDLE = { current: null, finished: false, finishedPaths: [], visits: {}, record: null };

function Demo() {
  const initial = new URLSearchParams(location.search).get("flow") === "large" ? "large" : "small";
  const [flow, setFlow] = useState<"small" | "large">(initial);
  const [paused, setPaused] = useState(false);
  const [toast, setToast] = useState<string>();
  const describe = useMemo(() => (flow === "small" ? smallFlow() : largeFlow()), [flow]);
  const onRef = useRef<(m: ToUI) => void>(() => undefined);

  const send = (m: FromUI) => {
    switch (m.type) {
      case "reveal":
        setToast(`opening source for ${m.path}`);
        setTimeout(() => setToast(undefined), 2500);
        break;
      case "setControls":
        console.log("[demo] breakpoints/forces set", m.controls);
        break;
      case "draft":
        onRef.current({
          type: "draft",
          draft: {
            converted: [
              { target: "offer@term/cap_by_income", value: 120000, source: "set" },
              { target: "force@term/by_sector", value: 1, source: "force", row: null },
            ],
            dropped: [{ kind: "console_mutation", detail: "Python console mutations are not tracked; any such changes are not in this draft" }],
          },
        });
        break;
      case "run":
        setPaused(false);
        onRef.current({ type: "status", ...IDLE });
        break;
      case "debugScenario":
        setToast(`debugger run for ${m.label}${m.row == null ? "" : ` (record ${m.row})`}`);
        setTimeout(() => setToast(undefined), 2500);
        break;
      default:
        console.log("[demo] message to host", m);
    }
  };

  const listen = (on: (m: ToUI) => void) => {
    onRef.current = on;
    const t = setTimeout(() => on({ type: "describe", describe }), 0);
    return () => clearTimeout(t);
  };

  const pause = () => {
    if (flow !== "small") return;
    const { status, columns } = pausedRun();
    setPaused(true);
    onRef.current({ type: "status", ...status });
    onRef.current({ type: "state", columns, rows: 2, key: { name: "client_id", values: [1, 2] } });
  };

  return (
    <div className="demo">
      <div className="demo-bar">
        <button className={flow === "small" ? "active" : ""} onClick={() => setFlow("small")}>Small flow</button>
        <button className={flow === "large" ? "active" : ""} onClick={() => setFlow("large")}>Large flow ({largeFlow().size.calls} steps)</button>
        {flow === "small" && (
          <button className={paused ? "active" : ""} onClick={() => (paused ? send({ type: "run" }) : pause())}>
            {paused ? "Resume (end run)" : "Pause at risk_tree"}
          </button>
        )}
        {flow === "small" && (
          <button onClick={() => onRef.current({ type: "experiment", experiment: experimentResult() })}>Show experiment</button>
        )}
        <span style={{ marginLeft: "auto", alignSelf: "center", opacity: 0.7 }}>
          render harness — ?flow=large selects the big one
        </span>
      </div>
      <div className="demo-host">
        <App key={flow} send={send} listen={listen} can={can} />
      </div>
      {toast && <div className="demo-toast">{toast}</div>}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<Demo />);
