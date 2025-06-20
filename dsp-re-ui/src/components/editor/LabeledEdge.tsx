import React, { type FC, memo, useMemo } from 'react';
import {
  getSmoothStepPath,
  EdgeLabelRenderer,
  BaseEdge,
  type EdgeProps,
  Position,
} from '@xyflow/react';

import {type Edge} from "./types";
import { useNodes } from './EditorContext';
import {OP_SYMBOL_MAP} from "./util";
 
// this is a little helper component to render the actual edge label
function EdgeLabel({ transform, label }: { transform: string; label: string }) {
  return (
    <span
      style={{
        position: 'absolute',
        background: 'transparent',
        paddingLeft: 3,
        transform,
        fontSize: 10,
        transformOrigin: "left center",
      }}
      className="nodrag nopan text-sm"
    >
      {label}
    </span>
  );
}


const DEFAULT_SYMBOL = "∅"

// const getThresholdLabelList = (thresholds?: number[]) => {
//     if (!thresholds) { return [] }
//     if (thresholds.length === 1) { 
//         return [`>= ${thresholds[0]}`, `< ${thresholds[0]}`]
//     }
//     return thresholds.map((v,i)=>{
//         const v_n = thresholds[i+1];
//         if (v_n === undefined) { return `< ${thresholds[0]} | >= ${thresholds[thresholds.length-1]}`}
//         return `>= ${v} & < ${v_n}`

//     })
// }


const getThresholdLabelList = (thresholds?: number[]) => {
    if (!thresholds) { return [] }
    return [
        `< ${thresholds[0]}`,
        ...thresholds.map((v,i)=>{
            const v_n = thresholds[i+1];
            if (v_n === undefined) { return `≥ ${v}`}
            return `≥ ${v} & < ${v_n}`
        })
    ];
}


const LabeledEdge: FC<
  EdgeProps<Edge>
> = ({
    id,
    source,
    sourceX,
    sourceY,
    targetX,
    targetY,
    labelStyle,
    labelShowBg,
    labelBgStyle,
    labelBgPadding,
    labelBgBorderRadius,
    style,
    sourcePosition = Position.Bottom,
    targetPosition = Position.Top,
    markerEnd,
    markerStart,
    pathOptions,
    interactionWidth,
    data,
}) => {
  const [nodes] = useNodes();

  const nodeLabel = useMemo(()=>{
    const currNode = nodes.find((v)=>v.id == source);
    if (currNode == undefined) { return DEFAULT_SYMBOL }
    let labels: string[] = ["true", "false"];
    if (currNode.data.node_type === "numerical_test_node") {
        labels = [`${OP_SYMBOL_MAP[currNode.data.comparison_op]} ${currNode.data.threshold}`, "otherwise"];
    } else if (currNode.data.node_type === "categorical_test_node") {
        labels = [`{${currNode.data.category_list}}`, "otherwise"];
    } else if (currNode.data.node_type === "numerical_range_test_node") {
        const nodeData = currNode.data;
        labels = getThresholdLabelList(nodeData.thresholds);
    }
    return (data?.sourceIndex !== undefined && labels[data?.sourceIndex]) || DEFAULT_SYMBOL;
  }, [nodes, source, data])


  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    borderRadius: 0, // Make this square
    offset: pathOptions?.offset,
  });
 
  return (
    <>
    <BaseEdge
        id={id}
        path={edgePath}
        labelX={labelX}
        labelY={labelY}
        // label={`${data?.sourceIndex}`}
        labelStyle={labelStyle}
        labelShowBg={labelShowBg}
        labelBgStyle={labelBgStyle}
        labelBgPadding={labelBgPadding}
        labelBgBorderRadius={labelBgBorderRadius}
        style={style}
        markerEnd={markerEnd}
        markerStart={markerStart}
        interactionWidth={interactionWidth}
    />

    <EdgeLabelRenderer>
    {data?.sourceIndex != null  && (
      <EdgeLabel
        transform={`translate(0%, -100%) translate(${targetX}px,${targetY}px)`}
        label={nodeLabel}
      />
    )}
    </EdgeLabelRenderer>
  </>
  );
};
 
export default LabeledEdge;