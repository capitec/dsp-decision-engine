import React, { useMemo } from 'react';
import {
  getSmoothStepPath,
  EdgeLabelRenderer,
  BaseEdge,
  EdgeProps,
  Position,
  useNodes,
} from '@xyflow/react';
import { getNodeEdgeLabels, DEFAULT_SYMBOL } from '../../lib/node-utils';

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


const LabeledEdge: React.FC<EdgeProps> = ({
  id,
  source,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition = Position.Bottom,
  targetPosition = Position.Top,
  style = {},
  markerEnd,
  data,
}) => {
  const nodes = useNodes();

  const nodeLabel = useMemo(() => {
    const currNode = nodes.find((v) => v.id === source);
    if (currNode === undefined) { return DEFAULT_SYMBOL }

    // v2: variables are inline in node data as ParameterReference objects, no global map needed
    const labels = getNodeEdgeLabels(currNode.data, {});

    // Handle both single sourceIndex (number) and array of sourceIndex (number[])
    const sourceIndices = (data as any)?.sourceIndex;
    if (sourceIndices === undefined) {
      return DEFAULT_SYMBOL;
    }

    // If multiple source indices, show them as comma-separated labels
    if (Array.isArray(sourceIndices)) {
      if (sourceIndices.length === 0) {
        return DEFAULT_SYMBOL;
      }
      // Show labels for all source indices, joined with " / "
      const labelParts = sourceIndices
        .map(idx => labels[idx] || DEFAULT_SYMBOL)
        .filter((label, index, arr) => arr.indexOf(label) === index); // Remove duplicates
      return labelParts.join(' / ');
    }

    // Single source index
    return labels[sourceIndices] || DEFAULT_SYMBOL;
  }, [nodes, source, data]);

  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    borderRadius: 0,
  });

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          ...style,
          cursor: 'pointer',
        }}
        markerEnd={markerEnd}
      />
      {(data as any)?.sourceIndex !== undefined && nodeLabel && (
        <EdgeLabelRenderer>
        <EdgeLabel
          transform={`translate(0%, -100%) translate(${targetX}px,${targetY}px)`}
          label={nodeLabel}
        />
        </EdgeLabelRenderer>
      )}
    </>
  );
};

export default LabeledEdge;