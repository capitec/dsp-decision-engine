import { FunctionComponent, useMemo } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Node } from './types';
import { getLabel } from './util';
import { useFeatures } from './EditorContext';
 

export const ProcessNode: FunctionComponent<Pick<Node, "data" | "targetPosition"> & {isConnectable: boolean}> = ({
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
      <Handle
        type="source"
        position={Position.Bottom}
        id="output"
        style={{ background: 'output' }}
        isConnectable={isConnectable}
      />
    </>
  );
}