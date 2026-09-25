import { App, type EditorMessage } from "@decider/ui";
import { MainAreaWidget } from "@jupyterlab/apputils";
import { ReactWidget, ToolbarButton } from "@jupyterlab/ui-components";
import * as React from "react";
import type { Host } from "./host";

// Open source, Run to and Run through; the Python debugger, diffs and git revisions are VS Code's.
const CAN = new Set<EditorMessage>(["reveal", "runTo", "step"]);

/** The flow panel: decider's UI, with run controls in its toolbar since JupyterLab has no debug toolbar. */
export function flowPanel(host: Host, title: string, onError: (message: string) => void): MainAreaWidget {
  const content = ReactWidget.create(<App send={host.send} listen={host.listen} can={CAN} />);
  content.addClass("jp-DeciderPanel");
  const panel = new MainAreaWidget({ content });
  panel.id = `decider-${Math.random().toString(36).slice(2)}`;
  panel.title.label = `decider: ${title}`;
  panel.title.closable = true;
  const act = (f: () => Promise<void> | void) => () => void Promise.resolve().then(f).catch((e) => onError((e as Error).message));
  const buttons: [string, string, () => Promise<void> | void][] = [
    ["Start", "Start a debug run from the beginning, paused before the first step (again, to restart)", () => host.start()],
    ["Continue", "Run to the next breakpoint or the end", () => host.move("resume")],
    ["Step over", "Run the next step and pause after it", () => host.move("step")],
    ["Step in", "Pause at the next step, going into groups", () => host.move("step_into")],
    ["Step out", "Run to the end of the group the run is in", () => host.move("step_out")],
    ["Stop", "End the debug run", () => host.stop()],
  ];
  for (const [label, tooltip, f] of buttons) {
    panel.toolbar.addItem(label.toLowerCase().replace(" ", "-"), new ToolbarButton({ label, tooltip, onClick: act(f) }));
  }
  return panel;
}
