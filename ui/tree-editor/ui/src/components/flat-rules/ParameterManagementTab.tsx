/**
 * @deprecated ParameterManagementTab moved to shared-tree/ParameterManagementDialog
 * Use ParameterManagementDialog from '../shared-tree/ParameterManagementDialog' instead
 */

import React from 'react';

export const ParameterManagementTab: React.FC = () => null;

// Re-export for backwards compatibility
export type { Parameter, ParameterManagementDialogProps } from '../shared-tree/ParameterManagementDialog';
export { ParameterManagementDialog } from '../shared-tree/ParameterManagementDialog';
