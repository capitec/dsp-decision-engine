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
  Connection,
} from '@xyflow/react';
import axios from 'axios';
import {addNode, defaultLeafNode, formatNodes, linkNodeToParent} from './util'
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
    axios.get(`/api/projects/${projectId}/nodes`)
    .then(response=>formatNodes(response.data))
    .then(({nodes, edges, features})=> {
      setFeatures(features)
      setNodes((curr)=>nodes)
      setEdges((curr)=>edges)
    })
    .catch(error => {
      console.error(error);
    })
  }, [setNodes, setEdges])

  useEffect(() => {
    fetchProjectNodes(projectId)
  }, [projectId]);

 
  const onConnect = useCallback(
    ({ source, sourceHandle, target }: Edge | Connection) => setEdges(
      (edges) => linkNodeToParent(
          target,
          { parentNodeId: source, parentHandle: sourceHandle! },
          edges
        )),
    []
  );

  const onConnectEnd = useCallback<OnConnectEnd>(
    (event, connectionState) => {
      if (connectionState.isValid) { return }
      if (connectionState.fromHandle?.type === "target") {return}
      const { clientX, clientY } =
        'changedTouches' in event ? event.changedTouches[0] : event;

      setEdges((eds) => {
        const [newNodes, newEdges] = addNode(
          screenToFlowPosition({
            x: clientX,
            y: clientY,
          }),
          uuidv4(),
          defaultLeafNode,
          features,
          [],
          eds,
          {parentNodeId: connectionState.fromNode!.id, parentHandle: connectionState.fromHandle!.id!},
        )
        setNodes((nds) => [...nds, ...newNodes])
        return newEdges;
      })
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
