/**
 * Parameter management hook for tree editor
 * Handles CRUD operations on parameters stored in editor state
 */

import { useMemo } from 'react';
import { useEditorStore } from './useEditorStore';
import { extractParameterReferences } from './VariableActions';
import { Parameter } from '../shared-tree/ParameterManagementDialog';
import { ParameterOption } from '../shared-tree/ParameterizedNumericInput';
import { PolarsDataType } from '@/types/datasource-types';

export function useParameterManagement() {
  const parameters = useEditorStore((state) => state.parameters);
  const setParameters = useEditorStore((state) => state.setParameters);
  const nodes = useEditorStore((state) => state.nodes);

  // Extract vars referenced in nodes
  const variableNames = useMemo(() => extractParameterReferences(nodes), [nodes]);

  // Convert ParameterMap to Parameter[] for dialog
  const parametersForDialog = useMemo((): Parameter[] => {
    return Object.entries(parameters).map(([name, info]) => ({
      id: name,
      name: name,
      type: info.type as PolarsDataType,
      value: info.default_value
    }));
  }, [parameters]);

  // Convert to ParameterOption[] for input component
  const availableParameters = useMemo((): ParameterOption[] => {
    return Object.entries(parameters).map(([name, info]) => ({
      name,
      label: name,
      type: typeof info.type === 'string' ? info.type : 'String',
      value: info.default_value
    }));
  }, [parameters]);

  const createParameter = (name: string, type: PolarsDataType, value: any) => {
    setParameters({
      ...parameters,
      [name]: { type, default_value: value }
    });
  };

  const updateParameter = (oldName: string, updates: { name?: string; type?: PolarsDataType; value?: any }) => {
    const newParams = { ...parameters };
    const paramInfo = newParams[oldName];

    if (updates.name && updates.name !== oldName) {
      // Rename
      newParams[updates.name] = {
        type: updates.type ?? paramInfo.type,
        default_value: updates.value ?? paramInfo.default_value
      };
      delete newParams[oldName];
    } else {
      // Update in place
      newParams[oldName] = {
        type: updates.type ?? paramInfo.type,
        default_value: updates.value ?? paramInfo.default_value
      };
    }

    setParameters(newParams);
  };

  const deleteParameter = (name: string) => {
    const newParams = { ...parameters };
    delete newParams[name];
    setParameters(newParams);
  };

  const isParameterInUse = (name: string): boolean => {
    return variableNames.includes(name);
  };

  const saveParameters = (updatedParameters: Parameter[]) => {
    const newParams: Record<string, { type: PolarsDataType; default_value: any }> = {};
    updatedParameters.forEach(param => {
      newParams[param.name] = {
        type: param.type,
        default_value: param.value
      };
    });
    setParameters(newParams);
  };

  return {
    parameters,
    parametersForDialog,
    availableParameters,
    createParameter,
    updateParameter,
    deleteParameter,
    isParameterInUse,
    saveParameters,
    parameterCount: Object.keys(parameters).length
  };
}
