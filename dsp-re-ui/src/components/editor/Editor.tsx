import React, { useCallback, useEffect, useState } from 'react';

import {
  Background,
  ReactFlow,
  useNodesState,
  useEdgesState,
  addEdge,
  MiniMap,
  Controls,
} from '@xyflow/react';
import axios from 'axios';
import {formatNodes} from './util'
import {Node, Edge} from './types'
import {useFeatures, useEdges, useNodes} from './EditorContext'



export default function NodeEditor({projectId}: {projectId: string}) {
  const [nodes, setNodes, onNodesChange] = useNodes();
  const [edges, setEdges, onEdgesChange] = useEdges();
  const [features, setFeatures] = useFeatures();

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
 
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      fitView
      style={{ backgroundColor: "#F7F9FB" }}
    >
      <MiniMap />
      <Controls />
      <Background  />
    </ReactFlow>
  );
}
