import React, { useCallback, useMemo } from 'react';
import { useEditorStore } from '../../editor/useEditorStore';
import { Node, LeafNodeData } from '../../editor/types';
import { useToast } from "@/hooks/use-toast";
import { LeafNodeOutputEditor } from '@/components/shared-tree/LeafNodeOutputEditor';
import { formatOutputLabel } from '@/components/editor/util';

interface LeafNodeEditorProps {
  node: Node<LeafNodeData>;
}

/**
 * Leaf Node Editor - v2 only, modal-based editing.
 *
 * Features:
 * - Compact mode: dropdown + edit button in card wrapper
 * - Expanded mode: modal with field editors
 * - Save as New / Save to Existing with confirmation dialog
 */
export const LeafNodeEditor: React.FC<LeafNodeEditorProps> = ({ node }) => {
  const output = useEditorStore((state) => state.output);
  const nodes = useEditorStore((state) => state.nodes);
  const updateNodeData = useEditorStore((state) => state.updateNodeData);
  const setOutput = useEditorStore((state) => state.setOutput);
  const { toast } = useToast();

  const fieldNames = output.dtypes.map(([name]) => name);

  // Handle output selection change
  const handleOutputChange = useCallback((value: number) => {
    updateNodeData(node.id, { ...node.data, result_idx: value });
  }, [node.id, node.data, updateNodeData]);

  // Handle save as new output
  const handleSaveAsNew = useCallback((formData: Record<string, any>) => {
    const newData = [...output.data, formData];
    setOutput({
      ...output,
      data: newData,
    });

    // Update the node to point to the new output (last index)
    const newIndex = newData.length - 1;
    updateNodeData(node.id, { ...node.data, result_idx: newIndex });

    toast({
      title: "Saved as New Output",
      description: `Added new output record (index ${newIndex}) and updated leaf node.`,
    });
  }, [output, setOutput, updateNodeData, node.id, node.data, toast]);

  // Handle save to existing output
  const handleSaveToExisting = useCallback((formData: Record<string, any>) => {
    const currentIndex = node.data.result_idx ?? -1;

    if (currentIndex === -1) {
      // Update default
      setOutput({
        ...output,
        default: formData,
      });
      toast({
        title: "Updated Default Output",
        description: "Updated default output values.",
      });
    } else if (currentIndex >= 0 && currentIndex < output.data.length) {
      // Update specific record
      const newData = [...output.data];
      newData[currentIndex] = formData;
      setOutput({
        ...output,
        data: newData,
      });

      const affectedCount = nodes.filter(n =>
        n.data?.type === 'leaf' &&
        (n.data as any).result_idx === currentIndex
      ).length;

      toast({
        title: "Updated Output",
        description: `Updated output record at index ${currentIndex}. ${affectedCount} leaf node(s) affected.`,
      });
    }
  }, [output, setOutput, node.data.result_idx, nodes, toast]);

  // Count how many leaf nodes use the current output value
  const affectedNodesCount = useMemo(() => {
    const currentIndex = node.data.result_idx ?? -1;
    return nodes.filter(n =>
      n.data?.type === 'leaf' &&
      (n.data as any).result_idx === currentIndex
    ).length;
  }, [nodes, node.data.result_idx]);

  // Get output options for dropdown
  const outputOptions = useMemo(() => {
    const displayFmt = output.display_fmt;

    const options = [
      {
        value: -1,
        label: output.default
          ? formatOutputLabel(output.default, displayFmt, fieldNames)
          : 'Default Output',
        shortLabel: '-1',
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

  // Support both v2 (type) and v3 (node_type) formats
  const nodeType = (node.data as any).node_type || node.data.type;
  if (!node || nodeType !== 'leaf') {
    return null;
  }

  return (
    <LeafNodeOutputEditor
      leafValue={node.data.result_idx ?? -1}
      onLeafValueChange={handleOutputChange}
      output={output}
      outputOptions={outputOptions}
      onSaveAsNew={handleSaveAsNew}
      onSaveToExisting={handleSaveToExisting}
      showCompactCard={false}
      showUpdateConfirmation={true}
      affectedNodesCount={affectedNodesCount}
      showTooltips={true}
      maxScrollHeight="max-h-96"
      modalWidth="full"
      compactLabel="Leaf Node"
      compactInfoText={
        <>
          <p className="mb-1">Node Info:</p>
          <p>This is a terminal node (leaf) that returns a final decision/value.</p>
          <p className="mt-2 text-xs">result_idx={node.data.result_idx ?? -1}</p>
        </>
      }
    />
  );
};
