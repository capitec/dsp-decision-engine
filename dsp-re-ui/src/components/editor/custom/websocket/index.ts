import { BaseSchemes, NodeEditor, Root, Scope, ClassicPreset } from 'rete'
import { BaseArea, BaseAreaPlugin } from 'rete-area-plugin'
import { Preset } from "./presets/types"
import { AreaExtensions } from "rete-area-plugin";

export * as WebsocketPresets from './presets'

// import History from './history'
// import { Preset } from './presets/types'
// import type { Action } from './types'

// export type { Action as HistoryAction }
// export * as HistoryExtensions from './extensions'
// export * as Presets from './presets'
// export type { HistoryActions } from './presets/classic'
// export type { Preset } from './presets/types'

export interface Action {
}

/**
 * Websocket plugin enables interfacing with a websocket server
 */
export class WebsocketPlugin<Schemes extends BaseSchemes> extends Scope<never, [BaseArea<Schemes>, Root<Schemes>]> {
  private editor!: NodeEditor<Schemes>
  private area!: BaseAreaPlugin<Schemes, BaseArea<Schemes>>
  private presets: Preset<Schemes>[] = []

  constructor(props?: {  }) {
    super('websocket')

  }

  public addPreset(preset: Preset<Schemes>) {
    this.presets.push(preset as unknown as Preset<Schemes>)
    if (this.area && this.editor) (preset as unknown as Preset<Schemes>).connect(this)
  }

  setParent(scope: Scope<BaseArea<Schemes>, [Root<Schemes>]>): void {
    super.setParent(scope)

    this.area = this.parentScope<BaseAreaPlugin<Schemes, BaseArea<Schemes>>>(BaseAreaPlugin)
    this.editor = this.area.parentScope<NodeEditor<Schemes>>(NodeEditor)

    this.presets.forEach(preset => {
      preset.connect(this)
    })

    // setTimeout(async () => {
    //     // Mock loading of items
    //     // const socket = new ClassicPreset.Socket("socket");
    //     // const a = new ClassicPreset.Node("Override styles");
    //     // a.addOutput("a", new ClassicPreset.Output(socket));
    //     // a.addInput("a", new ClassicPreset.Input(socket));
    //     // await this.editor.addNode(a);
    
    //     // const b = new ClassicPreset.Node("Fully customized");
    //     // b.addOutput("a", new ClassicPreset.Output(socket));
    //     // b.addInput("a", new ClassicPreset.Input(socket));
    //     // await this.editor.addNode(b);
    
    //     // await this.area.translate(a.id, { x: 0, y: 0 });
    //     // await this.area.translate(b.id, { x: 300, y: 0 });
    
    //     // await this.editor.addConnection(new ClassicPreset.Connection(a, "a", b, "a"));
    //     //@ts-ignore
    //     AreaExtensions.zoomAt(this.area, this.editor.getNodes());
    // }, 10000)
  }

   /**
   * Destroy all views and remove all event listeners
   */
    public destroy() {
    }
}