import { FunctionComponent } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Node } from './types';
// import { LimitHandle } from './LimitHandle';
 

interface NodeProps {
    data: Node["data"], 
    isConnectable: boolean, 
    targetPosition: Position, 
    leftSourcePosition: Position, 
    rightSourcePosition: Position
}

export const LeftRightNode: FunctionComponent<Pick<Node, "data" | "targetPosition"> & {isConnectable: boolean}> = ({
  data,
  isConnectable,
  targetPosition = Position.Top,
}) => {
  return (
    <>
      <Handle type="target" position={targetPosition} isConnectable={isConnectable}/>
      {data?.label}
      <Handle id="left" type="source" position={Position.Left} isConnectable={isConnectable}/>
      <Handle id="right" type="source" position={Position.Right} isConnectable={isConnectable}/>
    </>
  );
}