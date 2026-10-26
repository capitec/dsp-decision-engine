import React from 'react';
import { NodeBlock } from './NodeBlock';
import type { RuleTableRow } from './types';
import { ParameterOption } from '../shared-tree/ParameterizedNumericInput';

interface TreeRendererProps {
  rootNode: RuleTableRow;
  treeId: string;
  startDepth?: number;
  availableParameters: ParameterOption[];
  onManageParameters?: () => void;
  onDragStart: (e: React.DragEvent, node: RuleTableRow, treeId: string) => void;
  onDragOver: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent, targetBranchId: string, treeId: string) => void;
  onCopyNode: (node: RuleTableRow) => void;
  onDeleteBranch: (treeId: string, branchId: string) => void;
  onAddBranch: (treeId: string, nodeId: string) => void;
  onAddOtherwiseBranch: (treeId: string, nodeId: string) => void;
  nodeErrors?: any[]; // Validation errors for this tree
}

export const TreeRenderer: React.FC<TreeRendererProps> = ({
  rootNode,
  treeId,
  startDepth = 0,
  availableParameters,
  onManageParameters,
  onDragStart,
  onDragOver,
  onDrop,
  onCopyNode,
  onDeleteBranch,
  onAddBranch,
  onAddOtherwiseBranch,
  nodeErrors = []
}) => {
  return (
    <NodeBlock
      node={rootNode}
      treeId={treeId}
      depth={0}
      absoluteDepth={startDepth}
      availableParameters={availableParameters}
      onManageParameters={onManageParameters}
      onDragStart={onDragStart}
      onDragOver={onDragOver}
      onDrop={onDrop}
      onCopyNode={onCopyNode}
      onDeleteBranch={onDeleteBranch}
      onAddBranch={onAddBranch}
      onAddOtherwiseBranch={onAddOtherwiseBranch}
      nodeErrors={nodeErrors}
      basePath={['rule']}
    />
  );
};