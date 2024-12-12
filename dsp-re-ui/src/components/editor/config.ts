import { Dispatch } from "react";
import { createRoot } from "react-dom/client";
import { NodeEditor, GetSchemes, ClassicPreset } from "rete";
import { AreaPlugin, AreaExtensions, Area2D } from "rete-area-plugin";
import {
  ConnectionPlugin,
  Presets as ConnectionPresets
} from "rete-connection-plugin";
import { ReactPlugin, Presets, ReactArea2D, Position } from "rete-react-plugin";
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



import { selectNode } from "@ctx/editor/editorActions";
import { EditorAction } from "@ctx/editor/editorTypes";
import { SelectorEntity } from "rete-area-plugin/_types/extensions/selectable";
import { ConnectionPathPlugin, Transformers } from "rete-connection-path-plugin";
import { curveStep, curveStepBefore, curveMonotoneX, curveLinear, CurveFactory, curveStepAfter } from "d3-shape";
import { ReadonlyPlugin } from "rete-readonly-plugin";


export class Node extends ClassicPreset.Node {
  width = 200;
  height = 40;
}

export class Connection<N extends Node> extends ClassicPreset.Connection<N, N> {
  curve?: CurveFactory;
}

export type Schemes = GetSchemes<
  Node,
  Connection<Node>
>;
export type AreaExtra = ReactArea2D<Schemes> | MinimapExtra | Area2D<Schemes>;

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


const removeEndOffset = (points: Position[]) => {
  // I really hate hardcoding the 12 here but it seems to be the way to do it from what i can see:
  // https://github.com/retejs/render-utils/blob/e9f163159b21e3c04cf70ec1d27873000d119927/src/sockets-position/dom-socket-position.ts#L39
  if (points.length !== 2) throw new Error('number of points should be equal to 2')
    const [st, end] = points;
    return [
      {x:st.x - 12, y:st.y},
      {x:end.x + 12, y:end.y},
    ]
}

// TODO consider implementing this as a CurveFactory
const extendDownward = (points: Position[]) => {
  if (points.length !== 2) throw new Error('number of points should be equal to 2')
  const [st, ed] = points;
  if (st.x === ed.x) {return points}
  const midpoint = Math.min((st.y+ed.y)/2, st.y+80)
  return [
    {x:st.x, y:st.y},
    {x:st.x, y: midpoint},
    {x:ed.x, y: midpoint},
    {x:ed.x, y:ed.y},
  ]
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
  const readonly = new ReadonlyPlugin<Schemes>();
  const applier = new ArrangeAppliers.TransitionApplier<Schemes, never>({
    duration: 500,
    timingFunction: (t) => t,
    async onTick() {
      await AreaExtensions.zoomAt(area, editor.getNodes());
    }
  });

  const pathPlugin = new ConnectionPathPlugin<Schemes, AreaExtra>({
    curve: (payload) => {
      return curveMonotoneX
    },
    transformer: () => (points) => extendDownward(removeEndOffset(points))
    // transformer: () => Transformers.classic({ vertical: true }),
    // arrow: () => true
  });

  // @ts-ignore
  const minimap = new MinimapPlugin<Schemes>({
    boundViewport: true
  });

  AreaExtensions.selectableNodes(area, selector, {
    // accumulating: AreaExtensions.accumulateOnCtrl()
    accumulating: {active: ()=>false}
  });



  render.addPreset(
    Presets.classic.setup({
      customize: {
        node(context) {
          return CustomNode;
        // //   if (context.payload.label === "Fully customized") {
        // //     return CustomNode;
        // //   }
        // //   if (context.payload.label === "Override styles") {
        // //     return StyledNode;
        // //   }
          // return Presets.classic.Node;
        },
        socket(context) {
          // return Presets.classic.Socket;
          return CustomSocket;
        },
        connection(context) {
          // return Presets.classic.Connection;
          return CustomConnection;
        }
      }
    })
  );
  render.addPreset(Presets.minimap.setup({ size: 200 }));
  connection.addPreset(ConnectionPresets.classic.setup());
  history.addPreset(HistoryPresets.classic.setup());
  arrange.addPreset(ArrangePresets.classic.setup());
  // websocketProvider.addPreset(WebsocketPresets.setup({}))
  // HistoryExtensions.keyboard(history);

  addCustomBackground(area);

  // editor.use(readonly.root);
  editor.use(area);
  area.use(readonly.area);
  area.use(history);
  // area.use(websocketProvider);
  area.use(render);
  area.use(minimap);
  area.use(arrange);
  render.use(pathPlugin);


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
      await arrange.layout({ 
        applier: animate ? applier : undefined,
        options: {
          'elk.direction': 'DOWN',
          'elk.algorithm': 'mrtree',
          'elk.alignment': 'DOWN',
          // @ts-ignore
          'elk.layered.spacing.nodeNodeBetweenLayers': 200,
          'elk.mrtree.weighting': 'CONSTRAINT',
          // @ts-ignore
          'elk.spacing.edgeNode': 0,
          // @ts-ignore
          'elk.spacing.nodeNode': 200,
        }
      });
      AreaExtensions.zoomAt(area, editor.getNodes());
    },
    setEditorDispatch: (d: Dispatch<EditorAction>) => {selector.editorDispatch = d},
    area,
    editor,
    socket,
    readonly,
  };
}
