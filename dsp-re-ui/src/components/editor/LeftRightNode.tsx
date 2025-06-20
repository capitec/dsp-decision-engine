import { FunctionComponent, useMemo } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Node } from './types';
import { getLabel } from './util';
import { useFeatures } from './EditorContext';
 

export const LeftRightNode: FunctionComponent<Pick<Node, "data" | "targetPosition"> & {isConnectable: boolean}> = ({
  data,
  isConnectable,
  targetPosition = Position.Top,
}) => {
  const [features, setFeatures] = useFeatures();
  const nodeLabel = useMemo(()=>getLabel(data,features), [features, data])

  return (
    <>
      <Handle type="target" position={targetPosition} isConnectable={isConnectable}/>
      {nodeLabel}
      {/* <Handle id="left" type="source" position={Position.Left} isConnectable={isConnectable}/>
      <Handle id="right" type="source" position={Position.Right} isConnectable={isConnectable}/> */}
      <Handle
        type="source"
        position={Position.Bottom}
        id="left"
        style={{ left: '15%', background: 'blue' }}
        isConnectable={isConnectable}
      />
      <Handle
          type="source"
          id="right"
          position={Position.Bottom}
          style={{ left: '85%', background: 'red' }}
          isConnectable={isConnectable}
        />
    </>
  );
}