import * as React from 'react'
import { ClassicScheme, RenderEmit, Presets } from "rete-react-plugin";
import styled, { css } from "styled-components";
import { $nodewidth, $socketmargin, $socketsize } from "./vars";
import { Trash2, PlusCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';

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

const onDelete = (id: string) => {
  console.log(id)
}
const onAddChild = (id: string) => {
  console.log(id)

}
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
        overflow-hidden
        ${selected ? 'ring-2 ring-primary/70' : ''}
      `}
      style={{
        width: `${width}px`,
        height: `${height}px`
      }}
      data-testid="node"
      key={id}
    >
      {/* Delete Button */}
      {selected && onDelete && (
        <Button 
          variant="destructive" 
          size="icon" 
          className="absolute top-1 left-1 z-10 w-6 h-6"
          onClick={() => onDelete(id)}
        >
          <Trash2 className="w-4 h-4" />
        </Button>
      )}

      {/* Add Child Button */}
      {selected && onAddChild && (
        <Button 
          variant="outline" 
          size="icon" 
          className="absolute top-1 right-1 z-10 w-6 h-6"
          onClick={() => onAddChild(id)}
        >
          <PlusCircle className="w-4 h-4" />
        </Button>
      )}

      {/* Input Sockets */}
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

      {/* Node Label */}
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

      {/* Output Sockets */}
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
