import * as React from "react";
import { ClassicScheme, RenderEmit, Presets } from "rete-react-plugin";
import styled, { css } from "styled-components";
import { $nodewidth, $socketmargin, $socketsize } from "./vars";

const { RefSocket, RefControl } = Presets.classic;

type NodeExtraData = { width?: number; height?: number };

// FAFAFC
// B8C3BD
// F0D97F
// EB6531
// 199CA2

export const NodeStyles = styled.div<
  NodeExtraData & { selected: boolean; styles?: (props: any) => any }
>`
  display: flex;
  background: #a4c2d5;
  border: 1px solid #B8C3BD;
  cursor: pointer;
  box-sizing: border-box;
  width: ${(props) =>
    Number.isFinite(props.width) ? `${props.width}px` : `${$nodewidth}px`};
  height: ${(props) =>
    Number.isFinite(props.height) ? `${props.height}px` : "auto"};
  padding-bottom: 6px;
  position: relative;
  user-select: none;
  &:hover {
    border: 2px solid #000;
  }
  ${(props) =>
    props.selected &&
    css`
      background: #199CA2;
    `}
  .title {
    display: flex;
    font-family: sans-serif;
    margin: auto 0px auto 10px;
    font-size: 16px;
    height: 100%;
    width: 100%;
    text-overflow: hidden;
    align-items: center;
  }
  .output {
  }
  .input {
  }
  .output-socket {
  }
  .input-socket {
    text-align: left;
    // margin-left: -1px;
    display: inline-block;
  }
  .input-title,
  .output-title {
  }
  .input-control {
  }
  .control {
  }
  ${(props) => props.styles && props.styles(props)}
`;

function sortByIndex<T extends [string, undefined | { index?: number }][]>(
  entries: T
) {
  entries.sort((a, b) => {
    const ai = a[1]?.index || 0;
    const bi = b[1]?.index || 0;

    return ai - bi;
  });
}

type Props<S extends ClassicScheme> = {
  data: S["Node"] & NodeExtraData;
  styles?: () => any;
  emit: RenderEmit<S>;
};
export type NodeComponent<Scheme extends ClassicScheme> = (
  props: Props<Scheme>
) => JSX.Element;

export function CustomNode<Scheme extends ClassicScheme>(props: Props<Scheme>) {
  const inputs = Object.entries(props.data.inputs);
  const outputs = Object.entries(props.data.outputs);
  const controls = Object.entries(props.data.controls);
  const selected = props.data.selected || false;
  const { id, label, width, height } = props.data;

  sortByIndex(inputs);
  sortByIndex(outputs);
  sortByIndex(controls);

  return (
    <NodeStyles
      selected={selected}
      width={width}
      height={height}
      styles={props.styles}
      data-testid="node"
    >

      <h4
        // onPointerDown={(e) => {
        //   e.stopPropagation();
        //   console.log(">>>");
        // }}
        className="title"
        data-testid="title"
      >
        {label}
      </h4>
      {/* Outputs */}
      {outputs.map(
        ([key, output]) =>
          output && (
              <RefSocket
                name="output-socket"
                side="output"
                emit={props.emit}
                socketKey={key}
                nodeId={id}
                payload={output.socket}
              />
          )
        )
      }

      {inputs.map(
        ([key, input]) =>
          input && (
              <RefSocket
                name="input-socket"
                emit={props.emit}
                side="input"
                socketKey={key}
                nodeId={id}
                payload={input.socket}
              />
          )
      )}

    </NodeStyles>
  );
}
