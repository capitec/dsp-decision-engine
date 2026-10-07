import React, { useState } from 'react';
import { useOnSelectionChange } from '@xyflow/react';
import { useEditorContext } from '../editor/EditorContext';
import { NoSelectionPanel } from './NoSelectionPanel';
import { EdgeSelectionPanel } from './EdgeSelectionPanel';
import { NodeSelectionPanel } from './NodeSelectionPanel';

interface SidebarProps {}

export const Sidebar: React.FC<SidebarProps> = () => {
  const { nodes, edges } = useEditorContext();
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  // Handle node and edge selection
  const onChange = React.useCallback(({ nodes: selectedNodes, edges: selectedEdges }: { nodes: any[], edges: any[] }) => {
    if (selectedNodes.length > 0) {
      setSelectedNodeId(selectedNodes[0].id);
      setSelectedEdgeId(null);
    } else if (selectedEdges.length > 0) {
      setSelectedEdgeId(selectedEdges[0].id);
      setSelectedNodeId(null);
    } else {
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
    }
  }, []);

  useOnSelectionChange({
    onChange,
  });

  // Get the selected node and edge
  const selectedNode = React.useMemo(() => {
    if (!selectedNodeId) return null;
    return nodes.find(n => n.id === selectedNodeId) || null;
  }, [nodes, selectedNodeId]);

  const selectedEdge = React.useMemo(() => {
    if (!selectedEdgeId) return null;
    return edges.find(e => e.id === selectedEdgeId) || null;
  }, [edges, selectedEdgeId]);

  return (
    <div className="w-80 bg-gray-50 border-l border-gray-200 h-full overflow-y-auto p-4 space-y-4">
      {selectedNode ? (
        <NodeSelectionPanel node={selectedNode} />
      ) : selectedEdge ? (
        <EdgeSelectionPanel edge={selectedEdge} />
      ) : (
        <NoSelectionPanel />
      )}
    </div>
  );
};