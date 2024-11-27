import { BaseSchemes } from 'rete'
import { WebsocketPlugin } from '..'


export type Preset<S extends BaseSchemes> = {
  connect: (history: WebsocketPlugin<S>) => void
}