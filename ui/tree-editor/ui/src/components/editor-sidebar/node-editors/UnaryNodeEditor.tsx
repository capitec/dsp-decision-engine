import React, { useState } from 'react';
import { useEditorStore } from '../../editor/useEditorStore';
import { UnaryConditionForm, UnaryOperatorSelector } from '@/components/shared-tree';
import { ParameterManagementDialog } from '@/components/shared-tree/ParameterManagementDialog';
import { Label } from "@/components/ui/label";
import type { UnaryCondition } from '@/components/flat-rules/types';
import { useShallow } from 'zustand/react/shallow';
import { useParameterManagement } from '../../editor/useParameterManagement';

interface UnaryNodeEditorProps {
  node: any;
}

/**
 * Sidebar editor for v3 unary nodes.
 * Note: then/otherwise branches are managed via EdgeSelectionPanel, not here.
 */
export const UnaryNodeEditor: React.FC<UnaryNodeEditorProps> = ({ node }) => {
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

  const condition = (node.data.condition || {
    op: '<=',
    feature: '',
    threshold: 0
  }) as UnaryCondition;

  return (
    <>
      <div className="space-y-2">
        {/* Operator Selector */}
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-600 text-sm w-20">Operator:</span>
          <UnaryOperatorSelector
            value={condition.op}
            onChange={(newOp) => {
              // Clear operator-specific fields when changing operators
              let newCondition: any = { ...condition, op: newOp };

              if (['<=', '<', '==', '>', '>=', '!='].includes(newOp)) {
                newCondition = { op: newOp, feature: condition.feature || '', threshold: (condition as any).threshold ?? 0 };
              } else if (newOp === 'between') {
                newCondition = { op: newOp, feature: condition.feature || '', min: null, max: null };
              } else if (newOp === 'isin') {
                newCondition = { op: newOp, feature: condition.feature || '', values: [] };
              } else if (newOp === 'string_match') {
                newCondition = { op: newOp, feature: condition.feature || '', patterns: [''], match_type: 'exact', case_sensitive: true, trim_whitespace: false };
              } else if (newOp === 'is_true' || newOp === 'is_false') {
                newCondition = { op: newOp, feature: condition.feature || '' };
              }

              updateNodeData(node.id, { condition: newCondition });
            }}
          />
        </div>

        <UnaryConditionForm
          condition={condition}
          onChange={(newCondition) => {
            updateNodeData(node.id, { condition: newCondition });
          }}
          availableFeatures={availableFeatures}
          availableParameters={availableParametersForInput}
          onManageParameters={() => setParametersDialogOpen(true)}
        />
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
