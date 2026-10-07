import React, { useCallback, useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useEditorStore } from '../../editor/useEditorStore';
import { ComputedFeatureInput } from '../../shared-tree/ComputedFeatureInput';
import {
  ComparisonOperatorSelector,
  getOperatorSymbol
} from '../../shared-tree/ComparisonOperatorSelector';
import { ParameterizedNumericInput } from '../../shared-tree/ParameterizedNumericInput';
import { ParameterManagementDialog } from '../../shared-tree/ParameterManagementDialog';
import { Node, UnaryNodeData } from '../../editor/types';
import { UnaryNodeSourceIndex } from '@/types/tree-v3-types';
import { isParamRef, makeParamRef, paramRefName } from '@/utils/param-refs';
import { useParameterManagement } from '../../editor/useParameterManagement';

interface NumericalTestNodeEditorProps {
  node: Node<UnaryNodeData>;
}


export const NumericalTestNodeEditor: React.FC<NumericalTestNodeEditorProps> = ({ node }) => {
  const edges = useEditorStore((state) => state.edges);
  const featureNames = useEditorStore(
    useShallow((state) => state.input_schema.map(([name]) => name))
  );
  const updateNodeDataStore = useEditorStore((state) => state.updateNodeData);

  const [parametersDialogOpen, setParametersDialogOpen] = useState(false);

  // Update node data
  const updateNodeData = useCallback((nodeId: string, updates: Partial<UnaryNodeData>) => {
    updateNodeDataStore(nodeId, updates);
  }, [updateNodeDataStore]);

  // Parameter management
  const {
    availableParameters,
    parametersForDialog,
    isParameterInUse,
    saveParameters
  } = useParameterManagement();

  if (!node || node.data.type !== 'unary') {
    return null;
  }

  const nodeData = node.data;
  const condition = nodeData.condition;
  const threshold = condition.threshold;

  // Get feature name for display
  const featureName = typeof condition.feature === 'string'
    ? condition.feature
    : (condition.feature as any)?.expression || 'unselected';

  return (
    <>
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Unary Test Node</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <label className="text-sm font-medium mb-2 block">Feature</label>
          <ComputedFeatureInput
            value={condition.feature}
            onChange={(newFeature) => updateNodeData(node.id, {
              ...nodeData,
              condition: { ...condition, feature: newFeature }
            })}
            availableFeatures={featureNames}
            availableParameters={availableParameters}
          />
        </div>

        <ComparisonOperatorSelector
          value={condition.op}
          onChange={(op) => updateNodeData(node.id, {
            ...nodeData,
            condition: { ...condition, op }
          })}
        />

        <ParameterizedNumericInput
          value={isParamRef(threshold) || typeof threshold === 'number' ? threshold : 0}
          onChange={(newThreshold) => {
            updateNodeData(node.id, {
              ...nodeData,
              condition: { ...condition, threshold: newThreshold }
            });
          }}
          availableParameters={availableParameters}
          label="Threshold"
          placeholder="0.0 (supports inf, nan)"
          allowInfinity={true}
          allowNaN={true}
          step="0.01"
          onManageParameters={() => setParametersDialogOpen(true)}
          showManageParameters={true}
        />

        <div className="pt-2 border-t">
          <div className="text-xs text-gray-600">
            <p className="font-medium mb-1">Condition Preview:</p>
            <div className="font-mono bg-gray-100 p-2 rounded">
              {`${featureName} ${getOperatorSymbol(condition.op)} `}
              {isParamRef(threshold) ? (
                <span className="text-blue-600 font-semibold">
                  ${paramRefName(threshold)}
                </span>
              ) : (
                <span>{typeof threshold === 'number' ? threshold : 0}</span>
              )}
            </div>
          </div>
        </div>

        <div className="pt-2 border-t">
          <div className="text-xs text-gray-500">
            <p className="mb-1">Edge Structure:</p>
            <p className="mb-1">• sourceIndex={UnaryNodeSourceIndex.THEN} (0) → then branch</p>
            <p className="mb-1">• sourceIndex={UnaryNodeSourceIndex.OTHERWISE} (1) → otherwise branch</p>
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
                    const label = sourceIdx === UnaryNodeSourceIndex.THEN ? 'then' : 'otherwise';
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

    {/* Parameter Management Dialog */}
    <ParameterManagementDialog
      open={parametersDialogOpen}
      onOpenChange={setParametersDialogOpen}
      parameters={parametersForDialog}
      onSave={saveParameters}
      isParameterInUse={isParameterInUse}
      title="Manage Variables"
      description="Define parameters with types and default values for use in nodes"
    />
    </>
  );
};
