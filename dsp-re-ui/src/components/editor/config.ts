import { Dispatch } from "react";
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
// import * as CustomMinimapPreset from "./custom/minimap";
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

import { selectNode } from "@ctx/editor/editorActions";
import { EditorAction } from "@ctx/editor/editorTypes";
import { SelectorEntity } from "rete-area-plugin/_types/extensions/selectable";

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

class CustomSelector<E extends SelectorEntity> extends AreaExtensions.Selector<E> {
  editorDispatch: Dispatch<EditorAction> | null = null;
  lastPicked: string | null = null
  constructor() {super();}

  pick(entity: Pick<E, 'label' | 'id'>) {
    this.lastPicked = this.pickId
    if (super.isPicked(entity)) {
      super.release();
      super.remove(entity);
      this.editorDispatch && this.editorDispatch(selectNode(null))
    } else {
      super.pick(entity);
      this.editorDispatch && this.editorDispatch(selectNode(entity.id))
    }
  }
  unselectAll(): void {
    super.unselectAll();
    // This is a bit hacky but for some reason this even gets called after all picks
    if (this.lastPicked === this.pickId) {
      super.release();
      this.editorDispatch && this.editorDispatch(selectNode(null))
    }
    this.lastPicked = this.pickId;
  }
}


export async function createEditor(container: HTMLElement) {
  let editorDispatch: Dispatch<EditorAction> | null = null;
  const socket = new ClassicPreset.Socket("socket");
  const history = new HistoryPlugin<Schemes, HistoryActions<Schemes>>();

  const editor = new NodeEditor<Schemes>();
  const area = new AreaPlugin<Schemes, AreaExtra>(container);
  const connection = new ConnectionPlugin<Schemes, AreaExtra>();
  const render = new ReactPlugin<Schemes, AreaExtra>({ createRoot });
  // @
  const arrange = new AutoArrangePlugin<Schemes>();
  const selector = new CustomSelector();
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

  AreaExtensions.selectableNodes(area, selector, {
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

  // area.addPipe(context => {
  //   if (!editorDispatch) return context
  //   if (!context || typeof context !== 'object' || !('type' in context)) return context

  //   if (context.type === 'nodepicked') {
  //       editorDispatch(selectNode(context.data.id))
  //   }

  //   return context
  // })


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
    setEditorDispatch: (d: Dispatch<EditorAction>) => {selector.editorDispatch = d},
    area,
    editor,
    socket,
  };
}
