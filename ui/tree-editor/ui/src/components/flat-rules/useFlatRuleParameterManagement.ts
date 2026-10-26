/**
 * Parameter management hook for flat-rules editor
 * Handles CRUD operations on parameters stored in flat-rule project state
 */

import { useMemo } from 'react';
import {
  useFlatRuleProject,
  useFlatRuleCreateParameter,
  useFlatRuleUpdateParameter,
  useFlatRuleDeleteParameter,
  useFlatRuleIsParameterInUse
} from './useFlatRuleStore';
import { Parameter } from '../shared-tree/ParameterManagementDialog';
import { ParameterOption } from '../shared-tree/ParameterizedNumericInput';
import { PolarsDataType } from '@/types/datasource-types';
import { FlatRuleParameter } from '../shared-tree/ParameterButton';

export function useFlatRuleParameterManagement() {
  const project = useFlatRuleProject();
  const createParameter = useFlatRuleCreateParameter();
  const updateParameter = useFlatRuleUpdateParameter();
  const deleteParameter = useFlatRuleDeleteParameter();
  const isParameterInUse = useFlatRuleIsParameterInUse();

  // Extract parameters from project data
  const parameters = useMemo(() => {
    const params = project.project_data?.parameters || {};
    return Object.entries(params).map(([name, paramInfo]): FlatRuleParameter => ({
      id: name,
      name: name,
      parameter_type: paramInfo.type,
      value: paramInfo.default_value
    }));
  }, [project.project_data?.parameters]);

  // Convert to Parameter[] for dialog
  const parametersForDialog = useMemo((): Parameter[] => {
    return parameters.map(p => ({
      id: p.id,
      name: p.name,
      type: p.parameter_type,
      value: p.value
    }));
  }, [parameters]);

  // Convert to ParameterOption[] for input component
  const availableParameters = useMemo((): ParameterOption[] => {
    return parameters.map(p => ({
      name: p.name,
      label: p.name,
      type: typeof p.parameter_type === 'string' ? p.parameter_type : 'String',
      value: p.value
    }));
  }, [parameters]);

  const saveParameters = (updatedParameters: Parameter[]) => {
    // Build the new parameters object
    const newParams: Record<string, { type: PolarsDataType; default_value: any }> = {};
    updatedParameters.forEach(param => {
      newParams[param.name] = {
        type: param.type,
        default_value: param.value
      };
    });

    // Replace all parameters at once (batch update)
    // Note: This assumes there's a bulk update action in the store
    // For now, we'll delete all and recreate
    const oldParams = project.project_data?.parameters || {};
    Object.keys(oldParams).forEach(name => {
      deleteParameter(name);
    });
    updatedParameters.forEach(param => {
      createParameter(param.name, param.type, param.value);
    });
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
    parameterCount: parameters.length
  };
}
