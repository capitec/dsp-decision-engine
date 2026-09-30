import { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { App, type EditorMessage } from "../src/App";
import type { FromUI, ToUI } from "../src/model/protocol";
import "../src/style.css";
import { largeFlow, smallFlow } from "./mock";
import "./theme.css";

// A mock host: delivers one describe payload, and prints the messages the UI would send the
// extension (opening source) instead of acting on them.
const can = new Set<EditorMessage>(["reveal"]);

function Demo() {
  const initial = new URLSearchParams(location.search).get("flow") === "large" ? "large" : "small";
  const [flow, setFlow] = useState<"small" | "large">(initial);
  const [toast, setToast] = useState<string>();
  const describe = useMemo(() => (flow === "small" ? smallFlow() : largeFlow()), [flow]);

  const send = (m: FromUI) => {
    if (m.type === "reveal") {
      setToast(`opening source for ${m.path}`);
      setTimeout(() => setToast(undefined), 2500);
    } else {
      console.log("[demo] message to host", m);
    }
  };
  const listen = (on: (m: ToUI) => void) => {
    const t = setTimeout(() => on({ type: "describe", describe }), 0);
    return () => clearTimeout(t);
  };

  return (
    <div className="demo">
      <div className="demo-bar">
        <button className={flow === "small" ? "active" : ""} onClick={() => setFlow("small")}>Small flow</button>
        <button className={flow === "large" ? "active" : ""} onClick={() => setFlow("large")}>Large flow ({largeFlow().size.calls} steps)</button>
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
