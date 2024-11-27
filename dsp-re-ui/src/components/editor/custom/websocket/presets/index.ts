import { BaseSchemes, ConnectionId, NodeEditor, NodeId } from 'rete'
import { BaseArea, BaseAreaPlugin } from 'rete-area-plugin'
import { Preset } from "./types"

import { WebsocketPlugin } from '..'


function trackNodes<S extends BaseSchemes>(history: WebsocketPlugin<S>, props: { }) {
  const area = history.parentScope<BaseAreaPlugin<S, BaseArea<S>>>(BaseAreaPlugin)
  const editor = area.parentScope<NodeEditor<S>>(NodeEditor)

  //@ts-ignore
  editor.addPipe(context => {
    if (context.type === 'nodecreated') {
        console.log({"node created": context})
    }
    if (context.type === 'noderemoved') {
        console.log({"node removed": context})
    }
    return context
  })
  //@ts-ignore
  area.addPipe(context => {
    if (!context || typeof context !== 'object' || !('type' in context)) return context

    if (context.type === 'nodepicked') {
        console.log({"node picked": context})
    } else if (context.type === 'nodedragged') {
        console.log({"node dragged": context})
    } else if (context.type === 'nodetranslated') {
        console.log({"node translated": context})
    }

    return context
  })
}

function trackConnections<S extends BaseSchemes>(history: WebsocketPlugin<S>) {
  const editor = history.parentScope().parentScope<NodeEditor<S>>(NodeEditor)
  //@ts-ignore
  editor.addPipe(context => {
    if (context.type === 'connectioncreated') {
        console.log({"connection created": context})
    } else if (context.type === 'connectionremoved') {
    //   const connection = connections.get(context.data.id)
      console.log({"connection removed": context})
    }

    return context
  })
}



export function setup<S extends BaseSchemes>(props?: { }): Preset<S> {
    return {
      connect(websocket) {
        // const timing = props?.timing ?? history.timing * 2
  
        trackNodes(websocket, { })
        trackConnections(websocket)
      }
    }
  }