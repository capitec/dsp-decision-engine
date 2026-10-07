/**
 * Shared parameter/variable types used across tree editor and flat-rules
 */

import { PolarsDataType, PolarsType } from '@/types/datasource-types';

/**
 * Generic parameter definition used in parameter management
 */
export interface Parameter {
  id: string;
  name: string;
  type: PolarsDataType;
  value: any;
}

/**
 * Legacy type alias for flat-rules compatibility
 * @deprecated Use Parameter instead
 */
export interface FlatRuleParameter {
  id: string;
  name: string;
  parameter_type: PolarsDataType;
  value: any;
}

/**
 * Convert FlatRuleParameter to Parameter format
 */
export function toParameter(param: FlatRuleParameter): Parameter {
  return {
    id: param.id,
    name: param.name,
    type: param.parameter_type,
    value: param.value
  };
}

/**
 * Convert Parameter to FlatRuleParameter format
 */
export function toFlatRuleParameter(param: Parameter): FlatRuleParameter {
  return {
    id: param.id,
    name: param.name,
    parameter_type: param.type,
    value: param.value
  };
}
