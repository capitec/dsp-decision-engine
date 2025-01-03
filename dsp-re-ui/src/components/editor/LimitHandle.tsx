import React from 'react';
import { Handle, HandleProps, useHandleConnections } from '@xyflow/react';
 
export const LimitHandle = (props: HandleProps & {connectionCount: number}) => {
  const connections = useHandleConnections({
    type: props.type,
  });
 
  return (
    <Handle
      {...props}
      isConnectable={props.isConnectable && (connections.length < props.connectionCount)}
    />
  );
};
 