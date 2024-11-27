import { BaseSchemes } from "rete";
import { HistoryActions, HistoryPlugin } from "rete-history-plugin";


export function keyboard<S extends BaseSchemes, A extends HistoryActions<S>>(
  container: HTMLElement,
  history: HistoryPlugin<S, A>,
) {
  document.addEventListener("keydown", (e) => {

    if (!e.ctrlKey && !e.metaKey) return;

    switch (e.code) {
      case "KeyZ":
        if (e.shiftKey) {
          void history.redo();
        } else {
          void history.undo();
        }
        break;
      default:
    }
  });
}
