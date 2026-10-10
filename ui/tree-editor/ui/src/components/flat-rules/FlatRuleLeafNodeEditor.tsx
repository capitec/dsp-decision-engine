import React, { useCallback, useMemo } from 'react';
import { useFlatRuleProject, useFlatRuleUpdateNodeValue, useFlatRuleSaveAsNewOutput, useFlatRuleSaveToExistingOutput } from './useFlatRuleStore';
import { RuleTableRow } from './types';
import { useToast } from "@/hooks/use-toast";
import { LeafNodeOutputEditor } from '@/components/shared-tree/LeafNodeOutputEditor';
import { formatOutputLabel } from '@/components/editor/util';
import type { TreeOutput } from '@/types/tree-types';

interface FlatRuleLeafNodeEditorProps {
  node: RuleTableRow;
  treeId: string;
}

/**
 * Flat Rule Leaf Node Editor - uses shared LeafNodeOutputEditor component
 *
 * Features:
 * - Compact mode: inline dropdown + edit button (no card wrapper)
 * - Expanded mode: modal with field editors
 * - Save as New / Save to Existing (no confirmation dialog)
 */
export const FlatRuleLeafNodeEditor: React.FC<FlatRuleLeafNodeEditorProps> = ({
  node,
  treeId
}) => {
  const project = useFlatRuleProject();
  const updateNodeValue = useFlatRuleUpdateNodeValue();
  const saveAsNewOutput = useFlatRuleSaveAsNewOutput();
  const saveToExistingOutput = useFlatRuleSaveToExistingOutput();
  const { toast } = useToast();

  // Get output configuration from project (nested under output field)
  const output: TreeOutput = useMemo(() => {
    return {
      dtypes: project.project_data?.output?.dtypes || [],
      type_defs: project.project_data?.output?.type_defs || {},
      data: project.project_data?.output?.data || [],
      default: project.project_data?.output?.default || null,
      display_fmt: project.project_data?.output?.display_fmt
    };
  }, [project.project_data]);

  const fieldNames = output.dtypes.map(([name]) => name);

  // Handle output selection change
  const handleOutputChange = useCallback((value: number) => {
    updateNodeValue(treeId, node.id, 'result_idx', value);
  }, [treeId, node.id, updateNodeValue]);

  // Handle save as new output
  const handleSaveAsNew = useCallback((formData: Record<string, any>) => {
    saveAsNewOutput(formData);

    // Update the node to point to the new output (last index)
    const newIndex = output.data.length; // Will be the index after adding
    updateNodeValue(treeId, node.id, 'result_idx', newIndex);

    toast({
      title: "Saved as New Output",
      description: `Added new output record (index ${newIndex}) and updated leaf node.`,
    });
  }, [saveAsNewOutput, updateNodeValue, treeId, node.id, output.data.length, toast]);

  // Handle save to existing
  const handleSaveToExisting = useCallback((formData: Record<string, any>) => {
    const currentIndex = node.type === 'leaf' ? (node.result_idx ?? -1) : -1;
    saveToExistingOutput(currentIndex, formData);

    toast({
      title: "Saved to Existing Output",
      description: currentIndex === -1
        ? "Updated default output values."
        : `Updated output record at index ${currentIndex}.`,
    });
  }, [saveToExistingOutput, node, toast]);

  // Generate output options using display format
  const outputOptionsForSelect = useMemo(() => {
    const displayFmt = output.display_fmt;

    const options = [
      {
        value: -1,
        label: output.default
          ? formatOutputLabel(output.default, displayFmt, fieldNames)
          : 'Default Output',
        shortLabel: 'default',
      },
      ...output.data.map((record, index) => {
        const label = formatOutputLabel(record, displayFmt, fieldNames);

        return {
          value: index,
          label,
          shortLabel: String(index),
        };
      }),
    ];

    return options;
  }, [output.data, output.default, output.display_fmt, fieldNames]);

  if (node.type !== 'leaf') {
    return null;
  }

  return (
    <div className="contents" data-testid="leaf-output">
    <LeafNodeOutputEditor
      leafValue={node.result_idx ?? -1}
      onLeafValueChange={handleOutputChange}
      output={output}
      outputOptions={outputOptionsForSelect}
      onSaveAsNew={handleSaveAsNew}
      onSaveToExisting={handleSaveToExisting}
      showCompactCard={false}
      showCompactLabel={false}
      showUpdateConfirmation={false}
      showTooltips={false}
      maxScrollHeight="max-h-60"
      modalWidth="fixed"
    />
    </div>
  );
};