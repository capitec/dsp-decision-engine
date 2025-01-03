import { FunctionComponent } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Node } from './types';
import { LimitHandle } from './LimitHandle';
 

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
      <LimitHandle type="target" position={targetPosition} isConnectable={isConnectable} connectionCount={1}/>
      {data?.label}
      <LimitHandle id="left" type="source" position={Position.Left} isConnectable={isConnectable} connectionCount={1}/>
      <LimitHandle id="right" type="source" position={Position.Right} isConnectable={isConnectable} connectionCount={1}/>
    </>
  );
}