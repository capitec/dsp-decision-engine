import React, {
  createContext,
  useContext,
  useCallback,
  useMemo,
  useEffect,
} from "react";
import {
  OnNodesChange,
  OnEdgesChange,
  useReactFlow,
  OnConnectEnd,
  Connection,
} from "@xyflow/react";
import { useShallow } from 'zustand/react/shallow';
import { Node, Edge } from "./types";
import { useEditorStore } from './useEditorStore';
import { formatTree, detectSubtrees } from '../editor/util';
import { getNextEdge, createsCycle } from './util';

// XYFlow-specific context (only hooks that need ReactFlow context)
// NOTE: handleRearrange is included here but should NOT be spread onto ReactFlow component
// (it's not a valid ReactFlow prop and will cause React warnings)
interface XYFlowContextState {
  onNodesChange: OnNodesChange<Node>;
  onEdgesChange: OnEdgesChange<Edge>;
  onConnect: (params: Connection | Edge) => void;
  onConnectEnd: OnConnectEnd;
  handleRearrange: () => void;
}

export const XYFlowContext = createContext<XYFlowContextState>({
  onNodesChange: () => {},
  onEdgesChange: () => {},
  onConnect: () => {},
  onConnectEnd: () => {},
  handleRearrange: () => {},
});

const ChangeTypeIgnoreList = new Set(["select"]);

interface EditorProviderProps {
  children: React.ReactNode;
  projectId: string;
}

export const EditorProvider: React.FC<EditorProviderProps> = ({
  children,
  projectId,
}) => {
  const { screenToFlowPosition } = useReactFlow();

  // Get all state and actions from Zustand store
  const nodes = useEditorStore((state) => state.nodes);
  const edges = useEditorStore((state) => state.edges);
  const subtrees = useEditorStore((state) => state.subtrees);
  const autoRearrange = useEditorStore((state) => state.autoRearrange);

  const setNodes = useEditorStore((state) => state.setNodes);
  const setEdges = useEditorStore((state) => state.setEdges);
  const setSubtrees = useEditorStore((state) => state.setSubtrees);
  const setIsDirty = useEditorStore((state) => state.setIsDirty);
  const loadProject = useEditorStore((state) => state.loadProject);
  const addNode = useEditorStore((state) => state.addNode);
  const addEdge = useEditorStore((state) => state.addEdge);
  const removeEdge = useEditorStore((state) => state.removeEdge);

  // Load project on mount
  useEffect(() => {
    if (projectId) {
      loadProject(projectId);
    }
  }, [projectId, loadProject]);

  // XYFlow change handlers
  const onNodesChange: OnNodesChange<Node> = useCallback(
    (changes) => {
      const hasDirtyChanges = changes.some(change => !ChangeTypeIgnoreList.has(change.type));
      if (hasDirtyChanges) {
        setIsDirty(true);
      }

      // Apply changes to nodes
      setNodes((currentNodes) => {
        let updatedNodes = [...currentNodes];

        for (const change of changes) {
          if (change.type === 'position' && change.position) {
            updatedNodes = updatedNodes.map(node =>
              node.id === change.id ? { ...node, position: change.position! } : node
            );
          } else if (change.type === 'dimensions' && change.dimensions) {
            updatedNodes = updatedNodes.map(node =>
              node.id === change.id ? { ...node, measured: change.dimensions } : node
            );
          } else if (change.type === 'remove') {
            updatedNodes = updatedNodes.filter(node => node.id !== change.id);
          } else if (change.type === 'select') {
            updatedNodes = updatedNodes.map(node =>
              node.id === change.id ? { ...node, selected: change.selected } : node
            );
          }
        }

        return updatedNodes;
      });
    },
    [setNodes, setIsDirty]
  );

  const onEdgesChange: OnEdgesChange<Edge> = useCallback(
    (changes) => {
      const hasDirtyChanges = changes.some(change => !ChangeTypeIgnoreList.has(change.type));
      if (hasDirtyChanges) {
        setIsDirty(true);
      }

      // Apply changes to edges
      setEdges((currentEdges) => {
        let updatedEdges = [...currentEdges];

        for (const change of changes) {
          if (change.type === 'remove') {
            updatedEdges = updatedEdges.filter(edge => edge.id !== change.id);
          } else if (change.type === 'select') {
            updatedEdges = updatedEdges.map(edge =>
              edge.id === change.id ? { ...edge, selected: change.selected } : edge
            );
          }
        }

        return updatedEdges;
      });
    },
    [setEdges, setIsDirty]
  );

  // Connection handler
  const onConnect = useCallback(
    (params: Connection | Edge) => {
      const sourceIndex = getNextEdge(edges, params.source!);

      const newEdge: Edge = {
        id: `${params.source}:${params.target}`,
        source: params.source!,
        sourceHandle: params.sourceHandle || undefined,
        target: params.target!,
        targetHandle: params.targetHandle || undefined,
        type: "labeledEdge" as const,
        data: { sourceIndex: [sourceIndex] }, // v3 uses arrays
      };

      if (createsCycle(newEdge, edges)) return;

      // Remove any existing connection to the target
      const filteredEdges = edges.filter((edge) => edge.target !== params.target);
      setEdges([...filteredEdges, newEdge]);
    },
    [edges, setEdges]
  );

  // Handle drag-and-drop node creation
  const onConnectEnd: OnConnectEnd = useCallback(
    (event, connectionState) => {
      if (connectionState.isValid) {
        return;
      }

      if (connectionState.fromHandle?.type === "target") {
        return;
      }

      if (!connectionState?.fromNode || !connectionState?.fromHandle) {
        return;
      }

      const { clientX, clientY } = "changedTouches" in event ? event.changedTouches[0] : event;
      const position = screenToFlowPosition({ x: clientX, y: clientY });

      const sourceNodeId = connectionState.fromNode.id;
      const nodeId = crypto.randomUUID();

      // Create the node
      const newNode: Node = {
        id: nodeId,
        type: "default" as const,
        position,
        data: {
          type: "leaf" as const,
          result_idx: -1,
        },
      };

      addNode(newNode);

      // Create the edge
      const sourceIndex = getNextEdge(edges, sourceNodeId);
      const newEdge: Edge = {
        id: `${sourceNodeId}:${nodeId}`,
        source: sourceNodeId,
        target: nodeId,
        sourceHandle: "output",
        type: "labeledEdge" as const,
        data: { sourceIndex: [sourceIndex] }, // v3 uses arrays
      };

      addEdge(newEdge);
    },
    [screenToFlowPosition, edges, addNode, addEdge]
  );

  // Rearrange tree
  const handleRearrange = useCallback(() => {
    formatTree(nodes, edges).then(newNodes => {
      setNodes(newNodes);
    });
  }, [nodes, edges, setNodes]);

  // Auto-rearrange effect with debouncing
  useEffect(() => {
    if (autoRearrange && nodes.length > 0) {
      const timeoutId = setTimeout(() => {
        handleRearrange();
      }, 300);

      return () => clearTimeout(timeoutId);
    }
  }, [autoRearrange, nodes.length, handleRearrange]);

  // Auto-detect and update subtrees
  useEffect(() => {
    const newSubtrees = detectSubtrees(nodes, edges, subtrees);
    // setSubtrees marks the project dirty, so a freshly loaded project would look edited
    if (JSON.stringify(newSubtrees) !== JSON.stringify(subtrees)) setSubtrees(newSubtrees);
  }, [nodes, edges]);

  const contextValue: XYFlowContextState = useMemo(() => ({
    onNodesChange,
    onEdgesChange,
    onConnect,
    onConnectEnd,
    handleRearrange, // Keep in context for access, but don't spread onto ReactFlow
  }), [onNodesChange, onEdgesChange, onConnect, onConnectEnd, handleRearrange]);

  return (
    <XYFlowContext.Provider value={contextValue}>
      {children}
    </XYFlowContext.Provider>
  );
};

// Hook to get XYFlow-specific context
export const useXYFlowContext = () => {
  return useContext(XYFlowContext);
};

// Backward compatibility - hook to get full editor "context" (now from Zustand)
export const useEditorContext = () => {
  const xyFlowHooks = useXYFlowContext();

  // Get all state from Zustand store - use useShallow for arrays/objects
  const featureNames = useEditorStore(
    useShallow((state) => state.input_schema.map(([name]) => name))
  );
  const nodes = useEditorStore((state) => state.nodes);
  const edges = useEditorStore((state) => state.edges);
  const subtrees = useEditorStore((state) => state.subtrees);

  // v2 state
  const output = useEditorStore((state) => state.output);
  const parameters = useEditorStore((state) => state.parameters);

  // Project management state
  const loading = useEditorStore((state) => state.loading);
  const saving = useEditorStore((state) => state.saving);
  const error = useEditorStore((state) => state.error);
  const isDirty = useEditorStore((state) => state.isDirty);
  const projectId = useEditorStore((state) => state.projectId);
  const projectName = useEditorStore((state) => state.projectName);
  const projectDescription = useEditorStore((state) => state.projectDescription);
  const autoRearrange = useEditorStore((state) => state.autoRearrange);

  // Get all actions from Zustand store (actions are stable, no need to memoize)
  const setOutput = useEditorStore((state) => state.setOutput);
  const setParameters = useEditorStore((state) => state.setParameters);
  const setProjectName = useEditorStore((state) => state.setProjectName);
  const setProjectDescription = useEditorStore((state) => state.setProjectDescription);
  const setAutoRearrange = useEditorStore((state) => state.setAutoRearrange);
  const saveProject = useEditorStore((state) => state.saveProject);
  // validateProject removed - validation now happens automatically on save

  // Get handleRearrange separately (not part of xyFlowHooks to avoid spreading it onto ReactFlow)
  const handleRearrange = xyFlowHooks.handleRearrange;

  // Memoize the returned object to prevent unnecessary re-renders
  return useMemo(() => ({
    // Core editor data
    features: featureNames, // Backward compatibility: return feature names array
    nodes,
    edges,
    subtrees,

    // v2 state
    output,
    parameters,
    setOutput,
    setParameters,

    // XYFlow integration (safe to spread onto ReactFlow)
    xyFlowHooks,

    // Project management
    loading,
    saving,
    error,
    isDirty,
    projectId,
    projectName,
    projectDescription,
    setProjectName,
    setProjectDescription,
    autoRearrange,
    setAutoRearrange,
    handleRearrange, // Exposed separately, not part of xyFlowHooks spread
    saveProject,
  }), [
    featureNames, nodes, edges, subtrees,
    output, parameters, setOutput, setParameters,
    xyFlowHooks,
    loading, saving, error, isDirty, projectId, projectName, projectDescription,
    setProjectName, setProjectDescription, autoRearrange, setAutoRearrange,
    handleRearrange,
    saveProject,
  ]);
};

// Convenience hooks for specific data
export const useFeatures = () => useEditorStore(
  useShallow((state) => state.input_schema.map(([name]) => name))
);
export const useNodes = () => useEditorStore((state) => state.nodes);
export const useEdges = () => useEditorStore((state) => state.edges);
