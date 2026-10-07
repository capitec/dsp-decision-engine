import React, { useCallback, useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Trash2, Plus } from "lucide-react";
import { useEditorStore } from '../../editor/useEditorStore';
import { ComputedFeatureInput } from '../../shared-tree/ComputedFeatureInput';
import { Node, CasesStringMatchNodeData } from '../../editor/types';
import { TStringMatchType, getCasesNodeBranchSourceIndex, getCasesNodeOtherwiseSourceIndex } from '@/types/tree-v3-types';
import { useParameterManagement } from '../../editor/useParameterManagement';

interface StringMatchNodeEditorProps {
  node: Node<CasesStringMatchNodeData>;
}

export const StringMatchNodeEditor: React.FC<StringMatchNodeEditorProps> = ({ node }) => {
  const edges = useEditorStore((state) => state.edges);
  const featureNames = useEditorStore(
    useShallow((state) => state.input_schema.map(([name]) => name))
  );
  const updateNodeData = useEditorStore((state) => state.updateNodeData);
  const [currentPattern, setCurrentPattern] = useState('');
  const [matchType, setMatchType] = useState<TStringMatchType>('exact');
  const [caseSensitive, setCaseSensitive] = useState(true);

  const { availableParameters } = useParameterManagement();

  // Ref for scrolling to bottom of the patterns list
  const patternsListRef = React.useRef<HTMLDivElement>(null);

  if (!node || node.data.type !== 'cases' || node.data.op !== 'string_match') {
    return null;
  }

  const nodeData = node.data;
  const feature = typeof nodeData.feature === 'string' ? nodeData.feature : '';
  const conditions = nodeData.conditions || [];

  const addCurrentPattern = useCallback(() => {
    if (!currentPattern.trim()) return;

    const newCondition = {
      pattern: currentPattern,
      match_type: matchType,
      case_sensitive: caseSensitive
    };

    updateNodeData(node.id, {
      ...nodeData,
      conditions: [...conditions, newCondition]
    });

    // Scroll to bottom after adding
    setTimeout(() => {
      if (patternsListRef.current) {
        patternsListRef.current.scrollTop = patternsListRef.current.scrollHeight;
      }
    }, 10);

    setCurrentPattern('');
  }, [node.id, nodeData, conditions, currentPattern, matchType, caseSensitive, updateNodeData]);

  const removeCondition = useCallback((index: number) => {
    updateNodeData(node.id, {
      ...nodeData,
      conditions: conditions.filter((_, i) => i !== index)
    });
  }, [node.id, nodeData, conditions, updateNodeData]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Cases String Match Node</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <label className="text-sm font-medium mb-2 block">Feature</label>
          <ComputedFeatureInput
            value={feature}
            onChange={(newFeature: string | { type: 'computed'; expression: string }) => updateNodeData(node.id, { ...nodeData, feature: newFeature })}
            availableFeatures={featureNames}
            availableParameters={availableParameters}
          />
        </div>

        <div className="space-y-2">
          <Label>Pattern Conditions</Label>
          <p className="text-xs text-gray-500">Each condition creates a separate branch</p>

          {/* Current conditions list */}
          {conditions.length > 0 && (
            <div ref={patternsListRef} className="space-y-1 mb-3 max-h-48 overflow-y-auto">
              {conditions.map((cond, index) => (
                <div key={`cond-${index}`} className="flex items-center justify-between bg-gray-100 p-2 rounded">
                  <div className="flex-1 mr-2">
                    <div className="text-xs text-gray-500">sourceIndex={getCasesNodeBranchSourceIndex(index)}</div>
                    <div className="text-sm font-mono break-all">"{cond.pattern}"</div>
                    <div className="text-xs text-gray-600">
                      {cond.match_type} {cond.case_sensitive ? '(case-sensitive)' : '(case-insensitive)'}
                    </div>
                  </div>
                  <Button
                    onClick={() => removeCondition(index)}
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 p-0 text-gray-400 hover:text-red-500 hover:bg-red-50"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              ))}
            </div>
          )}

          {/* Add new condition */}
          <div className="border rounded p-2 space-y-2">
            <Label className="text-xs">Add New Condition</Label>

            <Select value={matchType} onValueChange={(v) => setMatchType(v as TStringMatchType)}>
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="exact">Exact Match</SelectItem>
                <SelectItem value="contains">Contains</SelectItem>
                <SelectItem value="starts_with">Starts With</SelectItem>
                <SelectItem value="ends_with">Ends With</SelectItem>
                <SelectItem value="regex">Regular Expression</SelectItem>
              </SelectContent>
            </Select>

            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="case-sensitive-new"
                checked={caseSensitive}
                onChange={(e) => setCaseSensitive(e.target.checked)}
                className="h-4 w-4"
              />
              <Label htmlFor="case-sensitive-new" className="text-xs">Case Sensitive</Label>
            </div>

            <div className="flex space-x-2">
              <Input
                value={currentPattern}
                onChange={(e) => setCurrentPattern(e.target.value)}
                placeholder="Pattern..."
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    addCurrentPattern();
                  }
                }}
                className="h-8 flex-1"
              />
              <Button
                onClick={addCurrentPattern}
                size="sm"
                className="h-8 px-2"
                disabled={!currentPattern}
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>

        <div className="pt-2 border-t">
          <div className="text-xs text-gray-500">
            <p className="mb-1">Edge Structure:</p>
            <p className="mb-1">• sourceIndex=0,1,2... → pattern condition branches</p>
            <p className="mb-1">• sourceIndex={getCasesNodeOtherwiseSourceIndex(conditions.length)} → otherwise branch</p>
            {edges.filter(e => e.source === node.id).length === 0 ? (
              <p className="italic mt-2">No outgoing edges</p>
            ) : (
              <ul className="list-disc list-inside mt-2">
                {edges
                  .filter(e => e.source === node.id)
                  .sort((a, b) => {
                    const aIdx = a.data?.sourceIndex?.[0] ?? 0;
                    const bIdx = b.data?.sourceIndex?.[0] ?? 0;
                    return aIdx - bIdx;
                  })
                  .map(edge => {
                    const sourceIdx = edge.data?.sourceIndex?.[0] ?? 0;
                    const label = sourceIdx < conditions.length ? `condition ${sourceIdx}` : 'otherwise';
                    return (
                      <li key={edge.id}>
                        sourceIndex={sourceIdx} ({label}) → {edge.target}
                      </li>
                    );
                  })}
              </ul>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
