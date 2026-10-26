import {
  Background,
  ReactFlow,
  MiniMap,
  Controls,
  BackgroundVariant,
} from '@xyflow/react';
import { nodeTypes } from './nodes';
import { edgeTypes } from './edges';
import { Sidebar } from '../editor-sidebar';
import { useEditorContext } from './EditorContext';

// Basic styles for ReactFlow
import '@xyflow/react/dist/style.css';

export default function Editor() {
  const {nodes, edges, xyFlowHooks} = useEditorContext();

  // Destructure to remove handleRearrange from props that go to ReactFlow
  const { handleRearrange: _unused, ...reactFlowProps } = xyFlowHooks;


  // // Load project data from initial state
  // useEffect(() => {
  //   console.log('Loading project:', projectId, 'with initial state:', initialState);
    
  //   if (initialState?.editor_data) {
  //     // Load all editor data from initial state (progressive save format)
  //     const { 
  //       nodes: initialNodes = [], 
  //       edges: initialEdges = [],
  //       features: initialFeatures = ['feature_0', 'feature_1', 'feature_2', 'feature_3'],
  //       treeOutput: initialTreeOutput = {
  //         data: [["Allow", "Default Action"], ["Alert", "A rule suggests that there is an alert"], ["Block", "This is more serious and must be blocked"]],
  //         columns: ["Action", "Description"],
  //         displayColumn: 0
  //       }
  //     } = initialState.editor_data;
      
  //     setNodes(initialNodes);
  //     setEdges(initialEdges);
  //     setFeatures(initialFeatures);
  //     setTreeOutput(initialTreeOutput);
  //   } else if (initialState && (initialState.nodes || initialState.edges || initialState.features)) {
  //     // Load editor data directly (imported project format)
  //     const { 
  //       nodes: initialNodes = [], 
  //       edges: initialEdges = [],
  //       features: initialFeatures = ['feature_0', 'feature_1', 'feature_2', 'feature_3'],
  //       treeOutput: initialTreeOutput = {
  //         data: [["Allow", "Default Action"], ["Alert", "A rule suggests that there is an alert"], ["Block", "This is more serious and must be blocked"]],
  //         columns: ["Action", "Description"],
  //         displayColumn: 0
  //       }
  //     } = initialState;
      
  //     setNodes(initialNodes);
  //     setEdges(initialEdges);
  //     setFeatures(initialFeatures);
  //     setTreeOutput(initialTreeOutput);
  //   } else {
  //     // Create a default node for new projects
  //     setNodes([
  //       {
  //         id: '1',
  //         type: 'processNode',
  //         position: { x: 250, y: 100 },
  //         data: { 
  //           node_type: 'numerical_test_node',
  //           split_feature_id: 0,
  //           comparison_op: '<=',
  //           threshold: 5,
  //           default_left: false
  //         },
  //       },
  //     ]);
  //     // Features and treeOutput will use defaults from EditorContext
  //   }
  // }, [projectId, initialState, setNodes, setEdges, setFeatures, setTreeOutput]);

  // // Handle force updates from parent (for import functionality)
  // useEffect(() => {
  //   if (onForceUpdate) {
  //     // Expose update functions to parent
  //     onForceUpdate({
  //       updateNodes: setNodes,
  //       updateEdges: setEdges,
  //       updateFeatures: setFeatures,
  //       updateTreeOutput: setTreeOutput
  //     });
  //   }
  // }, [onForceUpdate, setNodes, setEdges, setFeatures, setTreeOutput]);


  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex' }}>
      {/* Main Editor Area */}
      <div style={{ flex: 1, position: 'relative' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          defaultNodes={nodes}
          defaultEdges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
          {...reactFlowProps}
        >
          <Controls />
          <MiniMap />
          <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
        </ReactFlow>
      </div>
      
      {/* Sidebar */}
      <Sidebar />
    </div>
  );
}