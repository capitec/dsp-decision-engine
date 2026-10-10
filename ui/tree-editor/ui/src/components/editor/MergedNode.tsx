import React, { useMemo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import { useEditorStore } from './useEditorStore';
import { Node } from "./types";
import { paramRefName } from '@/utils/param-refs';
import { formatOutputLabel } from './util';

// Type guard for computed features
const isComputedFeature = (feature: any): feature is { type: 'computed'; expression: string } => {
  return feature && typeof feature === 'object' && feature.type === 'computed' && typeof feature.expression === 'string';
};

// Extract feature display string - handles both string features and computed features
const getFeatureDisplay = (feature: any): string => {
  if (!feature) return 'unselected';
  if (typeof feature === 'string') {
    return feature.length > 15 ? `${feature.substring(0, 15)}...` : feature;
  }
  if (isComputedFeature(feature)) {
    // Show f(x) icon for computed features with truncated expression
    const expr = feature.expression;
    const truncated = expr.length > 12 ? `${expr.substring(0, 12)}...` : expr;
    return `ƒ(${truncated})`;
  }
  return 'unselected';
};

// Helper to get parameter/variable name from either format
const getParameterName = (value: any): string | null => {
  return paramRefName(value);
};

const MergedNode: React.FC<NodeProps<Node>> = ({ data: nodeData, selected, id }) => {
  const output = useEditorStore((state) => state.output);
  const invalidNodeIds = useEditorStore((state) => state.invalidNodeIds);
  const hasError = invalidNodeIds.has(id);

  const labelContent = useMemo(() => {
    // Extract feature display string based on node type
    let feature = '';
    if ('feature' in nodeData && nodeData.feature) {
      // Feature can be string or computed feature object
      feature = getFeatureDisplay(nodeData.feature);
    } else if (nodeData.type === 'unary' && 'condition' in nodeData && nodeData.condition?.feature) {
      feature = getFeatureDisplay(nodeData.condition.feature);
    }

    // Handle v3 node types
    switch (nodeData.type) {
      case 'unary': {
        // V3 UnaryNode - has a condition with different operators
        const condition = (nodeData as any).condition;
        if (!condition) {
          return <span>Unary (no condition)</span>;
        }

        const op = condition.op;

        // Handle different unary operators
        if (op === '<=' || op === '<' || op === '==' || op === '>' || op === '>=' || op === '!=') {
          // Threshold-based comparison
          const threshold = condition.threshold;
          const paramName = getParameterName(threshold);
          const thresholdDisplay = paramName
            ? `#${paramName}`
            : String(threshold);
          return (
            <div className="flex items-center justify-center gap-1.5 min-w-0">
              <span className="font-mono text-[9px] bg-blue-100 px-1 py-0.5 rounded shrink-0">
                {feature}
              </span>
              <span className="truncate text-[10px]">
                {op} {thresholdDisplay}
              </span>
            </div>
          );
        } else if (op === 'between') {
          // Between operator
          const min = condition.min;
          const max = condition.max;
          const minParamName = getParameterName(min);
          const maxParamName = getParameterName(max);
          const minDisplay = minParamName ? `#${minParamName}` : String(min);
          const maxDisplay = maxParamName ? `#${maxParamName}` : String(max);
          return (
            <div className="flex items-center justify-center gap-1.5 min-w-0">
              <span className="font-mono text-[9px] bg-blue-100 px-1 py-0.5 rounded shrink-0">
                {feature}
              </span>
              <span className="truncate text-[10px]">
                {minDisplay} ≤ x &lt; {maxDisplay}
              </span>
            </div>
          );
        } else if (op === 'isin') {
          // IsIn operator (categorical)
          const values = condition.values || [];
          let categoryDisplay: string;
          const paramName = getParameterName(values);
          if (paramName) {
            categoryDisplay = `in [#${paramName}]`;
          } else if (Array.isArray(values)) {
            categoryDisplay = `in [${values.slice(0, 3).join(', ')}${values.length > 3 ? '...' : ''}]`;
          } else {
            categoryDisplay = `in [?]`;
          }
          return (
            <div className="flex items-center justify-center gap-1.5 min-w-0">
              <span className="font-mono text-[9px] bg-purple-100 px-1 py-0.5 rounded shrink-0">
                {feature}
              </span>
              <span className="truncate text-[10px]">{categoryDisplay}</span>
            </div>
          );
        } else if (op === 'string_match') {
          // String match operator
          const matchTypeMap: Record<string, string> = {
            'exact': 'exact',
            'starts_with': 'starts',
            'ends_with': 'ends',
            'contains': 'contains',
            'regex': 'regex'
          };
          const matchType = matchTypeMap[condition.match_type] || 'exact';
          const patternCount = condition.patterns?.length ?? 0;
          const caseIcon = !condition.case_sensitive ? 'i' : '';
          return (
            <div className="flex items-center justify-center gap-1.5 min-w-0">
              <span className="font-mono text-[9px] bg-yellow-100 px-1 py-0.5 rounded shrink-0">
                {feature}
              </span>
              <span className="truncate text-[10px]">
                {matchType}{caseIcon}({patternCount})
              </span>
            </div>
          );
        }
        return <span>Unary ({op})</span>;
      }

      case 'cases': {
        // V3 Cases node - check op to determine subtype
        const casesNode = nodeData as any;
        const conditions = casesNode.conditions || [];

        if (casesNode.op === 'ranges') {
          // CasesRanges - multiple range conditions
          return (
            <div className="flex items-center justify-center gap-1.5 min-w-0">
              <span className="font-mono text-[9px] bg-green-100 px-1 py-0.5 rounded shrink-0">
                {feature}
              </span>
              <span className="truncate text-[10px]">ranges({conditions.length})</span>
            </div>
          );
        } else if (casesNode.op === 'string_match') {
          // CasesStringMatch - multiple string match conditions
          const matchTypeMap: Record<string, string> = {
            'exact': 'exact',
            'starts_with': 'starts',
            'ends_with': 'ends',
            'contains': 'contains',
            'regex': 'regex'
          };
          const matchType = matchTypeMap[casesNode.match_type] || 'exact';
          const caseIcon = !casesNode.case_sensitive ? 'i' : '';
          return (
            <div className="flex items-center justify-center gap-1.5 min-w-0">
              <span className="font-mono text-[9px] bg-yellow-100 px-1 py-0.5 rounded shrink-0">
                {feature}
              </span>
              <span className="truncate text-[10px]">
                {matchType}{caseIcon}({conditions.length})
              </span>
            </div>
          );
        } else if (casesNode.op === 'isin') {
          // CasesIsIn - multiple IsIn conditions
          return (
            <div className="flex items-center justify-center gap-1.5 min-w-0">
              <span className="font-mono text-[9px] bg-purple-100 px-1 py-0.5 rounded shrink-0">
                {feature}
              </span>
              <span className="truncate text-[10px]">categories({conditions.length})</span>
            </div>
          );
        }
        // Fallback for unknown op
        return <span>Cases ({casesNode.op})</span>;
      }

      case 'composite': {
        // V3 CompositeNode - AND/OR logic
        const op = (nodeData as any).op || 'and';
        const conditions = (nodeData as any).conditions || [];
        return (
          <div className="flex items-center justify-center gap-1.5 min-w-0">
            <span className="font-mono text-[9px] bg-orange-100 px-1 py-0.5 rounded uppercase shrink-0">
              {op}
            </span>
            <span className="truncate text-[10px]">({conditions.length} conditions)</span>
          </div>
        );
      }

      case 'leaf': {
        // V3 LeafNode - uses result_idx instead of leaf_value
        const resultIdx = (nodeData as any).result_idx ?? -1;
        const fieldNames = output.dtypes.map(([name]) => name);

        if (fieldNames.length === 0) {
          return <span>No Output</span>;
        }

        let badgeText: string;
        let outputRecord: Record<string, any> | null | undefined;

        if (resultIdx === -1) {
          // Using default output
          outputRecord = output.default;
          badgeText = '-1';
        } else if (resultIdx >= 0 && resultIdx < output.data.length) {
          // Using specific output record
          outputRecord = output.data[resultIdx];
          badgeText = String(resultIdx);
        } else {
          // Invalid result index
          return <span>Invalid: {resultIdx}</span>;
        }

        // Format output label using display_fmt
        let valueDisplay = '';
        if (outputRecord) {
          valueDisplay = formatOutputLabel(outputRecord, output.display_fmt, fieldNames);
        }

        return (
          <div className="flex items-center justify-center gap-1.5 min-w-0">
            <span className="font-mono text-[9px] bg-gray-100 px-1 py-0.5 rounded shrink-0">
              {badgeText}
            </span>
            <span className="truncate text-[10px]">{valueDisplay || badgeText}</span>
          </div>
        );
      }

      default:
        return <span>Unknown Node Type</span>;
    }
  }, [output, nodeData]);

  // Determine if node is a leaf (shouldn't have source handle)
  const isLeaf = nodeData.type === 'leaf';

  return (
    <>
      <Handle
        type="target"
        position={Position.Top}
        className={`w-3 h-3 ${hasError ? 'bg-red-500' : 'bg-blue-500'}`}
        isConnectable={true}
      />

      {hasError && (
        <div className="absolute -top-2 -left-2 w-4 h-4 bg-red-500 rounded-full flex items-center justify-center text-white text-[10px] font-bold">
          !
        </div>
      )}

      <div className="text-[10px] font-medium text-gray-900 overflow-hidden w-full">
        {labelContent}
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        id="output"
        // We have to use opacity because if we dont register this node when we start
        // then xyflow doesnt realise we can connect from here and changing from a leaf to another node type causes issues
        className={`w-3 h-3 ${hasError ? 'bg-red-500' : 'bg-green-500'} ${isLeaf ? 'opacity-0 pointer-events-none' : ''}`}
        isConnectable={!isLeaf}
      />
    </>
  );
};

export default MergedNode;
