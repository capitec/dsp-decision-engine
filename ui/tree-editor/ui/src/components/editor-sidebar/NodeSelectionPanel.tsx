import React, { useCallback } from 'react';
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useEditorStore } from '../editor/useEditorStore';
import { NumericalTestNodeEditor } from './node-editors/NumericalTestNodeEditor';
import { CategoricalTestNodeEditor } from './node-editors/CategoricalTestNodeEditor';
import { RangeTestNodeEditor } from './node-editors/RangeTestNodeEditor';
import { StringMatchNodeEditor } from './node-editors/StringMatchNodeEditor';
import { LeafNodeEditor } from './node-editors/LeafNodeEditor';
import { UnaryNodeEditor } from './node-editors/UnaryNodeEditor';
import { CompositeNodeEditor } from './node-editors/CompositeNodeEditor';
import { CasesNodeEditor } from './node-editors/CasesNodeEditor';

interface NodeSelectionPanelProps {
  node: any;
}

export const NodeSelectionPanel: React.FC<NodeSelectionPanelProps> = ({ node }) => {
  const updateNodeData = useEditorStore((state) => state.updateNodeData);

  const handleNodeTypeChange = useCallback((newType: string) => {
    // Initialize default data based on node type
    // Use 'type' field to match API format
    const baseUpdate: any = { type: newType };

    // Add default condition structures for v3 types
    if (newType === 'unary') {
      baseUpdate.condition = { op: '<=', feature: '', threshold: 0 };
    } else if (newType === 'composite') {
      baseUpdate.op = 'and';
      baseUpdate.conditions = [];
    } else if (newType === 'leaf') {
      baseUpdate.result_idx = -1;
    } else if (newType === 'cases') {
      // Default to cases with ranges op when selecting "cases"
      baseUpdate.type = 'cases';
      baseUpdate.op = 'ranges';
      baseUpdate.feature = '';
      baseUpdate.conditions = [];
      baseUpdate.branches = [];
    }

    updateNodeData(node.id, baseUpdate);
  }, [updateNodeData, node.id]);

  if (!node || !node.data) {
    return (
      <div className="p-4 text-center text-gray-500">
        <p>No valid node selected</p>
      </div>
    );
  }

  // Support both v3 API format (type) and internal format (node_type)
  const nodeType = node.data.type || node.data.node_type;

  // Map cases type to 'cases' for the main dropdown
  const mainType = nodeType === 'cases' ? 'cases' : nodeType;

  return (
    <>
      {/* Node Type Selector - Common for all nodes */}
      <div className="space-y-2 mb-4">
        <Label>Node Type</Label>
        <Select
          value={mainType}
          onValueChange={handleNodeTypeChange}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="leaf">Leaf Node</SelectItem>
            <SelectItem value="unary">Unary Condition</SelectItem>
            <SelectItem value="composite">Composite (AND/OR/NOT)</SelectItem>
            <SelectItem value="cases">Cases</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Node-specific editor */}
      {(() => {
        switch (nodeType) {
          // V3 types
          case 'leaf':
            return <LeafNodeEditor node={node} />;

          case 'unary':
            return <UnaryNodeEditor node={node} />;

          case 'composite':
            return <CompositeNodeEditor node={node} />;

          // V3 cases type
          case 'cases':
            return <CasesNodeEditor node={node} />;

          // V2 types (legacy)
          case 'numerical_test_node':
            return <NumericalTestNodeEditor node={node} />;

          case 'categorical_test_node':
            return <CategoricalTestNodeEditor node={node} />;

          case 'numerical_range_test_node':
            return <RangeTestNodeEditor node={node} />;

          case 'string_match_node':
            return <StringMatchNodeEditor node={node} />;

          default:
            return (
              <div className="p-4 text-center text-gray-500">
                <p>Unknown node type: {nodeType}</p>
                <p className="text-xs mt-1">Supported types: leaf, unary, composite</p>
                <p className="text-xs mt-1">Legacy types: numerical_test_node, categorical_test_node, numerical_range_test_node, string_match_node</p>
              </div>
            );
        }
      })()}
    </>
  );
};