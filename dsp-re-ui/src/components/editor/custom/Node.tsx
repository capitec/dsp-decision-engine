import * as React from 'react'
import { ClassicScheme, RenderEmit, Presets } from "rete-react-plugin";
import styled, { css } from "styled-components";
import { $nodewidth, $socketmargin, $socketsize } from "./vars";

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

const { RefSocket, RefControl } = Presets.classic;

export const $nodecolor = 'rgba(110,136,255,0.8)'
export const $nodecolorselected = '#ffd92c'
export const $socketcolor = '#96b38a'


type NodeExtraData = { width?: number, height?: number }


function sortByIndex<T extends [string, undefined | { index?: number }][]>(entries: T) {
  entries.sort((a, b) => {
    const ai = a[1]?.index || 0
    const bi = b[1]?.index || 0

    return ai - bi
  })
}

type Props<S extends ClassicScheme> = {
  data: S['Node'] & NodeExtraData
  styles?: () => any
  emit: RenderEmit<S>
}
export type NodeComponent<Scheme extends ClassicScheme> = (props: Props<Scheme>) => JSX.Element


export function CustomNode<Scheme extends ClassicScheme>(props: Props<Scheme>) {
  const inputs = Object.entries(props.data.inputs)
  const outputs = Object.entries(props.data.outputs)
  const selected = props.data.selected || false
  const { id, label, width, height } = props.data
  sortByIndex(inputs)
  sortByIndex(outputs)

  return (
    <div 
      className={`
        relative 
        bg-accent-foreground 
        text-accent
        rounded-lg 
        border 
        border-primary/50 
        shadow-sm
        flex
        flex-col
        items-center
        justify-center
        w-[${width}px]
        h-[${height}px]
        overflow-hidden
        ${selected ? 'ring-2 ring-primary/70' : ''}
      `}
      data-testid="node"
      key={id}
  >
      <div className="absolute top-0 left-0 right-0 flex items-center justify-center px-2 -translate-y-1/2">
      {inputs.map(([key, input]) => input && (
          <RefSocket
            name="input-socket"
            side="input"
            socketKey={key}
            nodeId={id}
            emit={props.emit}
            payload={input.socket}
            data-testid="input-socket"
            key={key}
          />
        ))}
      </div>
      <div 
        className="
          text-center 
          py-2 
          px-4 
          font-semibold 
        " 
        data-testid="title"
      >
        {label}
      </div>
      <div className="absolute bottom-0 left-0 right-0 flex items-center justify-center px-2 translate-y-1/2">
      {outputs.map(([key, output]) => output && (
          <RefSocket
            name="output-socket"
            side="output"
            socketKey={key}
            nodeId={id}
            emit={props.emit}
            payload={output.socket}
            data-testid="output-socket"
            key={key}
          />
        ))}
      </div>

  </div>
  )
}

export function CustomNode4<Scheme extends ClassicScheme>(props: Props<Scheme>) {
  const inputs = Object.entries(props.data.inputs)
  const outputs = Object.entries(props.data.outputs)
  const selected = props.data.selected || false
  const { id, label } = props.data

  sortByIndex(inputs)
  sortByIndex(outputs)

  return (
    <div 
      className={`
        relative 
        bg-accent-foreground 
        text-accent
        rounded-lg 
        border 
        border-primary/50 
        shadow-sm
        ${selected ? 'ring-2 ring-primary/70' : ''}
      `}
      data-testid="node"
      key={id}
    >
      {/* Input Sockets - Positioned evenly across top border */}
      <div className="absolute top-0 left-0 right-0 flex justify-between px-2 -translate-y-1/2">
        {inputs.map(([key, input]) => input && (
          <RefSocket
            name="input-socket"
            side="input"
            socketKey={key}
            nodeId={id}
            emit={props.emit}
            payload={input.socket}
            data-testid="input-socket"
            key={key}
          />
        ))}
      </div>

      {/* Node Title */}
      <div 
        className="
          text-center 
          py-2 
          px-4 
          font-semibold 
          border-b 
          border-primary/20
        " 
        data-testid="title"
      >
        {label}
      </div>

      {/* Output Sockets - Positioned evenly across bottom border */}
      <div className="absolute bottom-0 left-0 right-0 flex justify-between px-2 translate-y-1/2">
        {outputs.map(([key, output]) => output && (
          <RefSocket
            name="output-socket"
            side="output"
            socketKey={key}
            nodeId={id}
            emit={props.emit}
            payload={output.socket}
            data-testid="output-socket"
            key={key}
          />
        ))}
      </div>
    </div>
  )
}
