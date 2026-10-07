import React, { useCallback } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ChipInput } from "@/components/ui/chip-input";
import { useEditorStore } from '../../editor/useEditorStore';
import { ComputedFeatureInput } from '../../shared-tree/ComputedFeatureInput';
import { Node, CasesRangesNodeData } from '../../editor/types';
import { RangeEndLogic, getCasesNodeBranchSourceIndex, getCasesNodeOtherwiseSourceIndex } from '@/types/tree-v3-types';
import { useParameterManagement } from '../../editor/useParameterManagement';

interface RangeTestNodeEditorProps {
  node: Node<CasesRangesNodeData>;
}

export const RangeTestNodeEditor: React.FC<RangeTestNodeEditorProps> = ({ node }) => {
  const edges = useEditorStore((state) => state.edges);
  const featureNames = useEditorStore(
    useShallow((state) => state.input_schema.map(([name]) => name))
  );
  const updateNodeData = useEditorStore((state) => state.updateNodeData);

  const { availableParameters } = useParameterManagement();

  if (!node || node.data.type !== 'cases' || node.data.op !== 'ranges') {
    return null;
  }

  const nodeData = node.data;
  const feature = typeof nodeData.feature === 'string' ? nodeData.feature : '';
  const conditions = nodeData.conditions || [];

  // Extract thresholds from conditions for display
  const thresholds = conditions
    .flatMap(c => [c.min, c.max])
    .filter((t): t is number => t !== null)
    .filter((value, index, array) => array.indexOf(value) === index)
    .sort((a, b) => a - b);

  // Handle threshold changes - convert to v3 range conditions
  const handleThresholdsChange = useCallback((stringValues: string[]) => {
    const thresholdList = stringValues
      .map(s => parseFloat(s))
      .filter(n => !isNaN(n))
      .filter((value, index, array) => array.indexOf(value) === index)
      .sort((a, b) => a - b);

    // Convert thresholds to range conditions
    // For n thresholds, create n+1 conditions
    const newConditions = [];

    if (thresholdList.length === 0) {
      // No thresholds - single unbounded condition
      newConditions.push({ min: null, max: null });
    } else {
      // First range: (-inf, first]
      newConditions.push({ min: null, max: thresholdList[0], end_logic: 'lower_inclusive' as RangeEndLogic });

      // Middle ranges: (threshold[i], threshold[i+1]]
      for (let i = 0; i < thresholdList.length - 1; i++) {
        newConditions.push({ min: thresholdList[i], max: thresholdList[i + 1] });
      }

      // Last range: (last, +inf)
      newConditions.push({ min: thresholdList[thresholdList.length - 1], max: null });
    }

    updateNodeData(node.id, { ...nodeData, conditions: newConditions });
  }, [updateNodeData, node.id, nodeData]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Range Test Node</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <label className="text-sm font-medium mb-2 block">Split Feature</label>
          <ComputedFeatureInput
            value={nodeData.feature || ''}
            onChange={(feature: string | { type: 'computed'; expression: string }) => updateNodeData(node.id, { feature })}
            availableFeatures={featureNames}
            availableParameters={availableParameters}
          />
        </div>

        <div className="space-y-2">
          <Label>Thresholds</Label>
          <ChipInput
            values={thresholds.map(t => t.toString())}
            onChange={handleThresholdsChange}
            placeholder="Type numbers and press Enter"
            validator={(value) => !isNaN(parseFloat(value.trim()))}
            formatter={(value) => {
              const num = parseFloat(value.trim());
              return isNaN(num) ? value : num.toString();
            }}
            className="text-sm"
          />
        </div>

        <div className="space-y-2">
          <Label className="text-sm font-medium">End Logic</Label>
          <Select
            value={conditions[0]?.end_logic || 'lower_inclusive'}
            onValueChange={(value) => {
              // Update end_logic for all conditions
              const updatedConditions = conditions.map(c => ({
                ...c,
                end_logic: value as RangeEndLogic
              }));
              updateNodeData(node.id, { conditions: updatedConditions });
            }}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select end logic" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="lower_inclusive">Lower Inclusive (values ≤ first threshold go left)</SelectItem>
              <SelectItem value="upper_inclusive">Upper Inclusive (values ≥ last threshold go right)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Show explicit range bounds */}
        <div className="pt-2 border-t">
          <div className="text-xs text-gray-600">
            <p className="font-medium mb-1">Generated Ranges:</p>
            <div className="bg-gray-50 p-3 rounded text-xs space-y-1">
              {thresholds.length === 0 ? (
                <div className="font-mono italic text-gray-500">
                  No thresholds defined
                </div>
              ) : (
                (() => {
                  const sortedThresholds = [...thresholds].sort((a, b) => a - b);
                  const ranges = [];

                  // First range: (-inf, first_threshold]
                  ranges.push(`(-\u221E, ${sortedThresholds[0]}]`);

                  // Middle ranges
                  for (let i = 0; i < sortedThresholds.length - 1; i++) {
                    ranges.push(`(${sortedThresholds[i]}, ${sortedThresholds[i + 1]}]`);
                  }

                  // Last range: (last_threshold, +inf)
                  ranges.push(`(${sortedThresholds[sortedThresholds.length - 1]}, +\u221E)`);

                  return ranges.map((range, index) => (
                    <div key={index} className="font-mono">
                      Branch {index}: {range}
                    </div>
                  ));
                })()
              )}
            </div>
          </div>
        </div>

        <div className="pt-2 border-t">
          <div className="text-xs text-gray-500">
            <p className="mb-1">Connected Edges:</p>
            {edges.filter(e => e.source === node.id).length === 0 ? (
              <p className="italic">No outgoing edges</p>
            ) : (
              <ul className="list-disc list-inside">
                {edges
                  .filter(e => e.source === node.id)
                  .sort((a, b) => ((a.data as any)?.sourceIndex || 0) - ((b.data as any)?.sourceIndex || 0))
                  .map(edge => (
                    <li key={edge.id}>
                      Branch {(edge.data as any)?.sourceIndex || 0} → {edge.target}
                    </li>
                  ))}
              </ul>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
