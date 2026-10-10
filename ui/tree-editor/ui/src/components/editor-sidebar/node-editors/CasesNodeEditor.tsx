import React, { useState } from 'react';
import { useEditorStore } from '../../editor/useEditorStore';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import { CasesConditionEditor } from '@/components/shared-tree/CasesConditionEditor';
import { ComputedFeatureInput } from '@/components/shared-tree/ComputedFeatureInput';
import { ParameterManagementDialog } from '@/components/shared-tree/ParameterManagementDialog';
import { useShallow } from 'zustand/react/shallow';
import { useParameterManagement } from '../../editor/useParameterManagement';

interface CasesNodeEditorProps {
  node: any;
}

/**
 * Sidebar editor for v3 cases nodes.
 * Shows case type selector and type-specific configuration.
 * Note: branch management happens via edge connections, not here.
 */
export const CasesNodeEditor: React.FC<CasesNodeEditorProps> = ({ node }) => {
  const updateNodeData = useEditorStore((state) => state.updateNodeData);
  const availableFeatures = useEditorStore(
    useShallow((state) => state.input_schema.map(([name]) => name))
  );
  const [parametersDialogOpen, setParametersDialogOpen] = useState(false);
  const [strictModeConfirmOpen, setStrictModeConfirmOpen] = useState(false);

  // Parameter management
  const {
    availableParameters: availableParametersForInput,
    parametersForDialog,
    isParameterInUse,
    saveParameters
  } = useParameterManagement();

  const conditions = node.data.conditions || [];

  // Check if enabling strict mode would modify conditions
  const wouldStrictModeModify = (conditions: any[]): boolean => {
    if (conditions.length === 0) return false;

    // Check if any lower bounds are set (not null)
    const hasLowerBounds = conditions.some((cond: any) => cond.min != null);

    // Check if conditions are already sorted by max
    const isSorted = conditions.every((cond: any, idx: number) => {
      if (idx === 0) return true;
      const prevMax = conditions[idx - 1].max ?? Infinity;
      const currMax = cond.max ?? Infinity;
      return prevMax <= currMax;
    });

    return hasLowerBounds || !isSorted;
  };

  // Apply strict mode changes
  const applyStrictMode = () => {
    const updatedConditions = conditions
      .map((cond: any) => ({ ...cond, min: null }))
      .sort((a: any, b: any) => {
        const aMax = a.max ?? Infinity;
        const bMax = b.max ?? Infinity;
        return aMax - bMax;
      });

    updateNodeData(node.id, {
      strict: true,
      conditions: updatedConditions
    });
  };

  return (
    <>
      <div className="space-y-2">
        {/* Case Type Selector */}
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-600 text-sm w-20">Case Type:</span>
          <Select
            value={node.data.op || 'ranges'}
            onValueChange={(newOp: 'ranges' | 'isin' | 'string_match') => {
              const baseUpdate: any = { type: 'cases', op: newOp, feature: node.data.feature || '' };

              // Initialize type-specific defaults
              if (newOp === 'string_match') {
                baseUpdate.match_type = node.data.match_type || 'exact';
                baseUpdate.case_sensitive = node.data.case_sensitive ?? true;
                baseUpdate.trim_whitespace = node.data.trim_whitespace ?? false;
              } else if (newOp === 'ranges') {
                baseUpdate.strict = node.data.strict ?? false;
              }

              updateNodeData(node.id, baseUpdate);
            }}
          >
            <SelectTrigger className="w-40 h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ranges">Ranges</SelectItem>
              <SelectItem value="isin">IsIn</SelectItem>
              <SelectItem value="string_match">String Match</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Feature Selector */}
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-600 text-sm w-20">Feature:</span>
          <div className="flex-1">
            <ComputedFeatureInput
              value={node.data.feature || ''}
              onChange={(value) => updateNodeData(node.id, { feature: value })}
              availableFeatures={availableFeatures}
              placeholder="feature or expression (e.g., feature1 + feature2)"
            />
          </div>
        </div>

        {/* String match settings */}
        {node.data.op === 'string_match' && (
          <>
            <div className="flex items-center gap-2">
              <span className="font-medium text-gray-600 text-sm w-20">Match Type:</span>
              <Select
                value={node.data.match_type || 'exact'}
                onValueChange={(value) => updateNodeData(node.id, { match_type: value })}
              >
                <SelectTrigger className="w-32 h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="exact">Exact</SelectItem>
                  <SelectItem value="starts_with">Starts With</SelectItem>
                  <SelectItem value="contains">Contains</SelectItem>
                  <SelectItem value="ends_with">Ends With</SelectItem>
                  <SelectItem value="regex">Regex</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-2 cursor-pointer">
                <Checkbox
                  checked={node.data.case_sensitive ?? true}
                  onCheckedChange={(checked) => updateNodeData(node.id, { case_sensitive: !!checked })}
                />
                <span className="text-sm text-gray-700">Case Sensitive</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <Checkbox
                  checked={node.data.trim_whitespace ?? false}
                  onCheckedChange={(checked) => updateNodeData(node.id, { trim_whitespace: !!checked })}
                />
                <span className="text-sm text-gray-700">Trim Whitespace</span>
              </label>
            </div>
          </>
        )}

        {/* Ranges strict mode */}
        {node.data.op === 'ranges' && (
          <div className="space-y-1">
            <label className="flex items-center gap-2 cursor-pointer">
              <Checkbox
                checked={node.data.strict ?? false}
                onCheckedChange={(checked) => {
                  const isEnablingStrictMode = !!checked;

                  if (isEnablingStrictMode && conditions.length > 0) {
                    // Check if enabling would modify conditions
                    if (wouldStrictModeModify(conditions)) {
                      // Show confirmation dialog
                      setStrictModeConfirmOpen(true);
                    } else {
                      // No changes needed, just enable
                      applyStrictMode();
                    }
                  } else {
                    // Just toggle off strict mode, keep conditions as-is
                    updateNodeData(node.id, { strict: false });
                  }
                }}
              />
              <span className="text-sm text-gray-700">Strict Mode</span>
            </label>
            <p className="text-xs text-gray-500 ml-6">
              {node.data.strict
                ? 'Lower bounds auto-inferred from previous max'
                : 'Enabling will clear lower bounds and sort by upper bound'}
            </p>
          </div>
        )}

        {/* Conditions editor */}
        <CasesConditionEditor
          caseType={`cases_${node.data.op}` as 'cases_ranges' | 'cases_isin' | 'cases_string_match'}
          conditions={conditions}
          onUpdate={(newConditions) => updateNodeData(node.id, { conditions: newConditions })}
          availableParameters={availableParametersForInput}
          onManageParameters={() => setParametersDialogOpen(true)}
          strictMode={node.data.op === 'ranges' ? (node.data.strict ?? false) : false}
        />
      </div>

      <ParameterManagementDialog
        open={parametersDialogOpen}
        onOpenChange={setParametersDialogOpen}
        parameters={parametersForDialog}
        onSave={saveParameters}
        isParameterInUse={isParameterInUse}
      />

      <AlertDialog open={strictModeConfirmOpen} onOpenChange={setStrictModeConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Enable Strict Mode?</AlertDialogTitle>
            <AlertDialogDescription>
              This operation will modify your conditions:
              <ul className="list-disc list-inside mt-2 space-y-1">
                <li>All lower bounds will be cleared</li>
                <li>Conditions will be sorted by upper bound</li>
                <li>Lower bounds will be auto-inferred from previous conditions</li>
              </ul>
              <p className="mt-2 font-semibold">This action cannot be undone.</p>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                applyStrictMode();
                setStrictModeConfirmOpen(false);
              }}
            >
              Enable Strict Mode
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
};
