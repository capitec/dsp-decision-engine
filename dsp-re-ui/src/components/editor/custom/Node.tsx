import * as React from 'react'
import { ClassicScheme, RenderEmit, Presets } from "rete-react-plugin";
import styled, { css } from "styled-components";
import { $nodewidth, $socketmargin, $socketsize } from "./vars";

const { RefSocket, RefControl } = Presets.classic;

export const $nodecolor = 'rgba(110,136,255,0.8)'
export const $nodecolorselected = '#ffd92c'
export const $socketcolor = '#96b38a'


type NodeExtraData = { width?: number, height?: number }

export const NodeStyles = styled.div<NodeExtraData & { selected: boolean, styles?: (props: any) => any }>`
    display: flex;
    flex-direction: column;
    justify-content: space-between; 
    align-items: center; 
    position: relative; 
    background: ${$nodecolor};
    border: 2px solid #4e58bf;
    border-radius: 10px;
    cursor: pointer;
    box-sizing: border-box;
    width: ${props => Number.isFinite(props.width)
    ? `${props.width}px`
    : `${$nodewidth}px`};
    height: ${props => Number.isFinite(props.height)
    ? `${props.height}px`
    : 'auto'};
    padding-bottom: 6px;
    user-select: none;
    line-height: initial;
    font-family: Arial;

    &:hover {
        background: lighten(${$nodecolor},4%);
    }
    ${props => props.selected && css`
        background: ${$nodecolorselected};
        border-color: #e3c000;
    `}
    .title {
        color: white;
        font-family: sans-serif;
        font-size: 18px;
        padding: 8px;
        text-align: center;
    }
    .output-socket {
      position: absolute;
      top: ${props => Number.isFinite(props.height)
        ? `${12+props.height!/2}px`
        : 'auto'};;
    }
    .input-socket {
      position: absolute;
      top: ${props => Number.isFinite(props.height)
        ? `-${-12+props.height!/2}px`
        : 'auto'};;
    }
    ${props => props.styles?.(props)}
`
// TODO some work is really needed on the top and bottom above to make it more dynamic

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
  const controls = Object.entries(props.data.controls)
  const selected = props.data.selected || false
  const { id, label, width, height } = props.data

  sortByIndex(inputs)
  sortByIndex(outputs)
  sortByIndex(controls)

  return (
    <NodeStyles
      selected={selected}
      width={width}
      height={height}
      styles={props.styles}
      data-testid="node"
    >

      {/* Inputs */}
      {inputs.map(([key, input]) => input && <RefSocket
          name="input-socket"
          side="input"
          socketKey={key}
          nodeId={id}
          emit={props.emit}
          payload={input.socket}
          data-testid="input-socket"
        />)}
      <div className="title" data-testid="title">{label}</div>
      {/* Outputs */}
      {outputs.map(([key, output]) => output && <RefSocket
          name="output-socket"
          side="output"
          socketKey={key}
          nodeId={id}
          emit={props.emit}
          payload={output.socket}
          data-testid="output-socket"
        />)}
    </NodeStyles>
  )
}