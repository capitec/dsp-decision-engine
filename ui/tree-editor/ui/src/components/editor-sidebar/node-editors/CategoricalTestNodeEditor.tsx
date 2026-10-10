import React, { useCallback, useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useEditorStore } from '../../editor/useEditorStore';
import { ComputedFeatureInput } from '../../shared-tree/ComputedFeatureInput';
import { ParameterizedCategoricalInput } from '../../shared-tree/ParameterizedCategoricalInput';
import { ParameterManagementDialog } from '../../shared-tree/ParameterManagementDialog';
import { Node, CasesIsInNodeData } from '../../editor/types';
import { isParamRef, paramRefName } from '@/utils/param-refs';
import { getCasesNodeBranchSourceIndex, getCasesNodeOtherwiseSourceIndex } from '@/types/tree-v3-types';
import { useParameterManagement } from '../../editor/useParameterManagement';

interface CategoricalTestNodeEditorProps {
  node: Node<CasesIsInNodeData>;
}

const isInputRef = isParamRef;

export const CategoricalTestNodeEditor: React.FC<CategoricalTestNodeEditorProps> = ({ node }) => {
  const edges = useEditorStore((state) => state.edges);
  const featureNames = useEditorStore(
    useShallow((state) => state.input_schema.map(([name]) => name))
  );
  const updateNodeDataStore = useEditorStore((state) => state.updateNodeData);

  const [parametersDialogOpen, setParametersDialogOpen] = useState(false);

  // Update node data
  const updateNodeData = useCallback((nodeId: string, updates: Partial<CasesIsInNodeData>) => {
    updateNodeDataStore(nodeId, updates);
  }, [updateNodeDataStore]);

  // Parameter management
  const {
    availableParameters,
    parametersForDialog,
    isParameterInUse,
    saveParameters
  } = useParameterManagement();

  if (!node || node.data.type !== 'cases' || node.data.op !== 'isin') {
    return null;
  }

  const nodeData = node.data;
  const feature = typeof nodeData.feature === 'string' ? nodeData.feature : '';
  const conditions = nodeData.conditions || [];

  return (
    <>
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Cases IsIn Node</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <label className="text-sm font-medium mb-2 block">Feature</label>
          <ComputedFeatureInput
            value={feature}
            onChange={(newFeature: string | { type: 'computed'; expression: string }) => updateNodeData(node.id, {
              ...nodeData,
              feature: newFeature
            })}
            availableFeatures={featureNames}
          />
        </div>

        <div className="space-y-2">
          <p className="text-sm font-medium">Conditions</p>
          <p className="text-xs text-gray-500">
            Each condition creates a branch. Values checked via IsIn operator.
          </p>
          {conditions.map((cond, idx) => {
            const values = cond.values;
            const hasInputRef = isInputRef(values);
            const displayValues = hasInputRef ? [] : (values as number[]);

            return (
              <div key={idx} className="border rounded p-2 space-y-2">
                <p className="text-xs font-medium">Condition {idx}</p>
                <ParameterizedCategoricalInput
                  value={values as any}
                  onChange={(newValues: any) => {
                    const newConditions = [...conditions];
                    newConditions[idx] = { values: newValues };
                    updateNodeData(node.id, { ...nodeData, conditions: newConditions });
                  }}
                  availableParameters={availableParameters}
                  label={`Values (sourceIndex=${getCasesNodeBranchSourceIndex(idx)})`}
                  placeholder="Type integers and press Enter"
                  onManageParameters={() => setParametersDialogOpen(true)}
                  showManageParameters={true}
                />
                <div className="font-mono text-xs bg-gray-100 p-1 rounded">
                  {hasInputRef ? (
                    <span className="text-blue-600 font-semibold">${paramRefName(values)}</span>
                  ) : displayValues.length === 0 ? (
                    <span className="italic text-gray-500">No values</span>
                  ) : (
                    `[${displayValues.join(', ')}]`
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <div className="pt-2 border-t">
          <div className="text-xs text-gray-500">
            <p className="mb-1">Edge Structure:</p>
            <p className="mb-1">• sourceIndex=0,1,2... → condition branches</p>
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
