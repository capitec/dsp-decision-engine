import React, { useCallback, useEffect, useState } from 'react';

import {
  Background,
  ReactFlow,
  useNodesState,
  useEdgesState,
  addEdge,
  MiniMap,
  Controls,
  useReactFlow,
  OnConnectEnd,
} from '@xyflow/react';
import axios from 'axios';
import {defaultLeafNode, formatNodes} from './util'
import {Node, Edge} from './types'
import {useFeatures, useEdges, useNodes} from './EditorContext'
import { nodeTypes } from './nodes';
import { v4 as uuidv4 } from 'uuid';



export default function NodeEditor({projectId}: {projectId: string}) {
  const [nodes, setNodes, onNodesChange] = useNodes();
  const [edges, setEdges, onEdgesChange] = useEdges();
  const [features, setFeatures] = useFeatures();
  const { screenToFlowPosition } = useReactFlow();

  const fetchProjectNodes = useCallback((projectId: string)=> {
    axios.get(`/api/projects/${projectId}/nodes`).then((response => {
      const {nodes, edges} = formatNodes(response.data)
      setFeatures(response.data.features)
      setNodes(nodes)
      setEdges(edges)
    })).catch(error => {
      console.error(error);
    })
  }, [setNodes, setEdges])

  useEffect(() => {
    fetchProjectNodes(projectId)
  }, [projectId]);

 
  const onConnect = useCallback(
    (params: any) => setEdges((els) => addEdge(params, els)),
    [],
  );

  const onConnectEnd = useCallback<OnConnectEnd>(
    (event, connectionState) => {
      // when a connection is dropped on the pane it's not valid
      if (!connectionState.isValid) {
        // we need to remove the wrapper bounds, in order to get the correct position
        const id = uuidv4();
        const { clientX, clientY } =
          'changedTouches' in event ? event.changedTouches[0] : event;
        const newNode: Node = {
          id,
          position: screenToFlowPosition({
            x: clientX,
            y: clientY,
          }),
          data: { label: `Node ${id}`, ...defaultLeafNode},
          origin: [0.5, 0.0],
        };
        
        setNodes((nds) => nds.concat(newNode));
        setEdges((eds) =>
          eds.concat({ id, source: connectionState.fromNode!.id, sourceHandle: connectionState.fromHandle!.id, target: id }),
        );
      }
    },
    [screenToFlowPosition],
  );
 
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onConnectEnd={onConnectEnd}
      nodeTypes={nodeTypes}
      fitView
      style={{ backgroundColor: "#F7F9FB" }}
    >
      <MiniMap />
      <Controls />
      <Background  />
    </ReactFlow>
  );
}
