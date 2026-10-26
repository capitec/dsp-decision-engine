import React, { useState } from 'react';
import { useEditorStore } from '../../editor/useEditorStore';
import { CompositeLogicSelector } from '@/components/shared-tree';
import { ParameterManagementDialog } from '@/components/shared-tree/ParameterManagementDialog';
import { CompositeConditionEditor } from '@/components/flat-rules/CompositeConditionEditor';
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";
import { useShallow } from 'zustand/react/shallow';
import { useParameterManagement } from '../../editor/useParameterManagement';

interface CompositeNodeEditorProps {
  node: any;
}

/**
 * Sidebar editor for v3 composite nodes (AND/OR/NOT).
 * Note: then/otherwise branches are managed via EdgeSelectionPanel, not here.
 */
export const CompositeNodeEditor: React.FC<CompositeNodeEditorProps> = ({ node }) => {
  const updateNodeData = useEditorStore((state) => state.updateNodeData);
  const availableFeatures = useEditorStore(
    useShallow((state) => state.input_schema.map(([name]) => name))
  );
  const [parametersDialogOpen, setParametersDialogOpen] = useState(false);

  // Parameter management
  const {
    availableParameters: availableParametersForInput,
    parametersForDialog,
    isParameterInUse,
    saveParameters
  } = useParameterManagement();

  const op = node.data.op || 'and';
  const conditions = node.data.conditions || [];

  return (
    <>
      <div className="space-y-4">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-sm font-medium text-gray-700">Logic Operator:</span>
          <CompositeLogicSelector
            value={op}
            conditionCount={conditions.length}
            onChange={(newLogic) => {
              updateNodeData(node.id, {
                op: newLogic
              });
            }}
          />
        </div>

        <div className="text-xs font-medium text-gray-600 mb-2">
          CONDITIONS ({op?.toUpperCase()})
        </div>

        {conditions.length === 0 && (
          <div className="text-xs text-red-600 p-2 bg-red-50 border border-red-200 rounded">
            A composite needs at least one condition. Add one below.
          </div>
        )}

        <CompositeConditionEditor
          conditions={conditions}
          onUpdate={(newConditions) => {
            updateNodeData(node.id, {
              conditions: newConditions
            });
          }}
          availableFeatures={availableFeatures}
          availableParameters={availableParametersForInput}
          onManageParameters={() => setParametersDialogOpen(true)}
          depth={0}
        />

        {/* Add condition button - NOT can only have 1 condition */}
        {(op !== 'not' || conditions.length === 0) && (
          <Button
            variant="outline"
            size="sm"
            className="h-8 text-xs border-dashed w-full"
            onClick={() => {
              const newCondition = {
                type: 'unary',
                op: '<=',
                feature: '',
                threshold: 0
              };
              const newConditions = [...conditions, newCondition];
              updateNodeData(node.id, {
                conditions: newConditions
              });
            }}
          >
            <Plus className="w-3 h-3 mr-1" />
            Add Condition
          </Button>
        )}
      </div>

      <ParameterManagementDialog
        open={parametersDialogOpen}
        onOpenChange={setParametersDialogOpen}
        parameters={parametersForDialog}
        onSave={saveParameters}
        isParameterInUse={isParameterInUse}
      />
    </>
  );
};
