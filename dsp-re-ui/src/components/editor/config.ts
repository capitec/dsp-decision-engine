import { createRoot } from "react-dom/client";
import { NodeEditor, GetSchemes, ClassicPreset } from "rete";
import { AreaPlugin, AreaExtensions } from "rete-area-plugin";
import {
  ConnectionPlugin,
  Presets as ConnectionPresets
} from "rete-connection-plugin";
import { ReactPlugin, Presets, ReactArea2D } from "rete-react-plugin";
import { MinimapExtra, MinimapPlugin } from "rete-minimap-plugin";
import { CustomNode } from "./custom/Node";
import { StyledNode } from "./custom/StyledNode";
import { CustomSocket } from "./custom/Socket";
import { CustomConnection } from "./custom/Connection";
import { addCustomBackground } from "./custom/background";
import { HistoryPlugin, HistoryActions, Presets as HistoryPresets, HistoryExtensions } from "rete-history-plugin";
import { WebsocketPlugin, WebsocketPresets } from "./custom/websocket";
import { keyboard } from "./custom/shortcuts"
import {
  AutoArrangePlugin,
  Presets as ArrangePresets,
  ArrangeAppliers
} from "rete-auto-arrange-plugin";

import {
  ContextMenuPlugin,
  Presets as ContextMenuPresets,
  ContextMenuExtra
} from "rete-context-menu-plugin";

export class Node extends ClassicPreset.Node {
  width = 180;
  height = 180;
}
export class Connection<N extends Node> extends ClassicPreset.Connection<N, N> {}

export type Schemes = GetSchemes<
  Node,
  Connection<Node>
>;
export type AreaExtra = ReactArea2D<Schemes> | MinimapExtra | ContextMenuExtra;

export async function createEditor(container: HTMLElement) {
  const socket = new ClassicPreset.Socket("socket");
  const history = new HistoryPlugin<Schemes, HistoryActions<Schemes>>();

  const editor = new NodeEditor<Schemes>();
  const area = new AreaPlugin<Schemes, AreaExtra>(container);
  const connection = new ConnectionPlugin<Schemes, AreaExtra>();
  const render = new ReactPlugin<Schemes, AreaExtra>({ createRoot });
  // @
  const arrange = new AutoArrangePlugin<Schemes>();
  const applier = new ArrangeAppliers.TransitionApplier<Schemes, never>({
    duration: 500,
    timingFunction: (t) => t,
    async onTick() {
      await AreaExtensions.zoomAt(area, editor.getNodes());
    }
  });

  // @ts-ignore
  const minimap = new MinimapPlugin<Schemes>({
    boundViewport: true
  });

  AreaExtensions.selectableNodes(area, AreaExtensions.selector(), {
    accumulating: AreaExtensions.accumulateOnCtrl()
  });

  const contextMenu = new ContextMenuPlugin<Schemes>({
    items: ContextMenuPresets.classic.setup([
      [
        "Node",
        () => {
          const a = new Node("C");
          a.addControl("a", new ClassicPreset.InputControl("text", {}));
          a.addInput("a", new ClassicPreset.Input(socket));
          a.addOutput("a", new ClassicPreset.Output(socket));
          setTimeout(() => {
            console.log(editor.getNodes());
          }, 100);
          return a;
        }
      ]
    ])
  });

  render.addPreset(
    Presets.classic.setup({
      customize: {
        // node(context) {
        // //   if (context.payload.label === "Fully customized") {
        // //     return CustomNode;
        // //   }
        // //   if (context.payload.label === "Override styles") {
        // //     return StyledNode;
        // //   }
        //   return Presets.classic.Node;
        // },
        socket(context) {
          return Presets.classic.Socket;
          // return CustomSocket;
        },
        connection(context) {
          return CustomConnection;
        }
      }
    })
  );
  render.addPreset(Presets.contextMenu.setup());
  render.addPreset(Presets.minimap.setup({ size: 200 }));
  connection.addPreset(ConnectionPresets.classic.setup());
  history.addPreset(HistoryPresets.classic.setup());
  arrange.addPreset(ArrangePresets.classic.setup());
  // websocketProvider.addPreset(WebsocketPresets.setup({}))
  // HistoryExtensions.keyboard(history);

  addCustomBackground(area);

  editor.use(area);
  area.use(connection);
  area.use(history);
  // area.use(websocketProvider);
  area.use(render);
  area.use(contextMenu);
  area.use(minimap);
  area.use(arrange);

  AreaExtensions.simpleNodesOrder(area);
  
  keyboard(
    container,
    history
  );
  // setTimeout(() => {
  //   AreaExtensions.zoomAt(area, editor.getNodes());
  // }, 100);

  return {
    destroy: () => area.destroy(),
    rearrangeLayout: async (animate: boolean = false) => {
      await arrange.layout({ applier: animate ? applier : undefined });
      AreaExtensions.zoomAt(area, editor.getNodes());
    },
    area,
    editor,
    socket,
  };
}
