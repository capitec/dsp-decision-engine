import { FunctionComponent, useMemo } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Node } from './types';
import { getLabel } from './util';
 

export const OutputNode: FunctionComponent<Pick<Node, "data" | "targetPosition"> & {isConnectable: boolean}> = ({
  data,
  isConnectable,
  targetPosition = Position.Top,
}) => {
  const nodeLabel = useMemo(()=>getLabel(data,[]), [data])
  return (
    <>
      <Handle type="target" position={targetPosition} isConnectable={isConnectable}/>
      {nodeLabel}
    </>
  );
}