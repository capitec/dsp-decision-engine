import type { UITypeDefs, OutputConfiguration } from './output-types';

export interface CategoryReference {
  fieldName: string;
  recordIndex: number;
  inArray: boolean; // true if in List/Set/Array, false if single value
}

export interface StructRecordReference {
  fieldName: string;
  recordIndex: number;
  inArray: boolean; // true if in List/Set/Array, false if single value
  displayValue: string; // The $key value
}

/**
 * Finds all references to a categorical value across output data
 */
export function findCategoryReferences(
  categoryValue: string,
  typeId: string,
  outputConfig: OutputConfiguration
): CategoryReference[] {
  const references: CategoryReference[] = [];

  // Check each record in output
  outputConfig.output.forEach((record, recordIndex) => {
    // Check each field
    outputConfig.dtypes.forEach(([fieldName, fieldType]) => {
      // Check if this field uses the categorical type
      const isDirectMatch =
        typeof fieldType === 'object' &&
        'type' in fieldType &&
        fieldType.type === 'Custom' &&
        'type_id' in fieldType &&
        (fieldType as any).type_id === typeId;

      if (isDirectMatch) {
        // Single categorical value
        if (record[fieldName] === categoryValue) {
          references.push({ fieldName, recordIndex, inArray: false });
        }
      }

      // Check if field is a List/Set/Array of this categorical type
      const isArrayMatch =
        typeof fieldType === 'object' &&
        'type' in fieldType &&
        (fieldType.type === 'List' || fieldType.type === 'Set' || fieldType.type === 'Array') &&
        'inner' in fieldType &&
        typeof fieldType.inner === 'object' &&
        'type' in fieldType.inner &&
        (fieldType.inner as any).type === 'Custom' &&
        'type_id' in fieldType.inner &&
        (fieldType.inner as any).type_id === typeId;

      if (isArrayMatch) {
        const arrayValue = record[fieldName];
        if (Array.isArray(arrayValue) && arrayValue.includes(categoryValue)) {
          references.push({ fieldName, recordIndex, inArray: true });
        }
      }
    });
  });

  return references;
}

/**
 * Finds all references to a struct record (by display value) across output data
 */
export function findStructRecordReferences(
  displayValue: string,
  typeId: string,
  outputConfig: OutputConfiguration
): StructRecordReference[] {
  const references: StructRecordReference[] = [];

  // Check each record in output
  outputConfig.output.forEach((record, recordIndex) => {
    // Check each field
    outputConfig.dtypes.forEach(([fieldName, fieldType]) => {
      // Check if this field uses the struct type
      const isDirectMatch =
        typeof fieldType === 'object' &&
        'type' in fieldType &&
        fieldType.type === 'Custom' &&
        'type_id' in fieldType &&
        (fieldType as any).type_id === typeId;

      if (isDirectMatch) {
        // Single struct reference
        const value = record[fieldName];
        if (value && typeof value === 'object' && '$key' in value && value.$key === displayValue) {
          references.push({ fieldName, recordIndex, inArray: false, displayValue });
        }
      }

      // Check if field is a List/Set/Array of this struct type
      const isArrayMatch =
        typeof fieldType === 'object' &&
        'type' in fieldType &&
        (fieldType.type === 'List' || fieldType.type === 'Set' || fieldType.type === 'Array') &&
        'inner' in fieldType &&
        typeof fieldType.inner === 'object' &&
        'type' in fieldType.inner &&
        (fieldType.inner as any).type === 'Custom' &&
        'type_id' in fieldType.inner &&
        (fieldType.inner as any).type_id === typeId;

      if (isArrayMatch) {
        const arrayValue = record[fieldName];
        if (Array.isArray(arrayValue)) {
          const hasReference = arrayValue.some(
            (item) => item && typeof item === 'object' && '$key' in item && item.$key === displayValue
          );
          if (hasReference) {
            references.push({ fieldName, recordIndex, inArray: true, displayValue });
          }
        }
      }
    });
  });

  return references;
}

/**
 * Removes all references to a categorical value from output data
 */
export function removeCategoryReferences(
  categoryValue: string,
  typeId: string,
  outputConfig: OutputConfiguration
): OutputConfiguration {
  const updatedOutput = outputConfig.output.map((record) => {
    const updatedRecord = { ...record };

    outputConfig.dtypes.forEach(([fieldName, fieldType]) => {
      // Check if this field uses the categorical type
      const isDirectMatch =
        typeof fieldType === 'object' &&
        'type' in fieldType &&
        fieldType.type === 'Custom' &&
        'type_id' in fieldType &&
        (fieldType as any).type_id === typeId;

      if (isDirectMatch && updatedRecord[fieldName] === categoryValue) {
        // Set to empty string or null
        updatedRecord[fieldName] = '';
      }

      // Check if field is a List/Set/Array of this categorical type
      const isArrayMatch =
        typeof fieldType === 'object' &&
        'type' in fieldType &&
        (fieldType.type === 'List' || fieldType.type === 'Set' || fieldType.type === 'Array') &&
        'inner' in fieldType &&
        typeof fieldType.inner === 'object' &&
        'type' in fieldType.inner &&
        (fieldType.inner as any).type === 'Custom' &&
        'type_id' in fieldType.inner &&
        (fieldType.inner as any).type_id === typeId;

      if (isArrayMatch && Array.isArray(updatedRecord[fieldName])) {
        // Filter out the category value
        updatedRecord[fieldName] = updatedRecord[fieldName].filter(
          (value: any) => value !== categoryValue
        );
      }
    });

    return updatedRecord;
  });

  return {
    ...outputConfig,
    output: updatedOutput
  };
}

/**
 * Removes all references to a struct record (by display value) from output data
 */
export function removeStructRecordReferences(
  displayValue: string,
  typeId: string,
  outputConfig: OutputConfiguration
): OutputConfiguration {
  const updatedOutput = outputConfig.output.map((record) => {
    const updatedRecord = { ...record };

    outputConfig.dtypes.forEach(([fieldName, fieldType]) => {
      // Check if this field uses the struct type
      const isDirectMatch =
        typeof fieldType === 'object' &&
        'type' in fieldType &&
        fieldType.type === 'Custom' &&
        'type_id' in fieldType &&
        (fieldType as any).type_id === typeId;

      if (isDirectMatch) {
        const value = updatedRecord[fieldName];
        if (value && typeof value === 'object' && '$key' in value && value.$key === displayValue) {
          // Set to null
          updatedRecord[fieldName] = null;
        }
      }

      // Check if field is a List/Set/Array of this struct type
      const isArrayMatch =
        typeof fieldType === 'object' &&
        'type' in fieldType &&
        (fieldType.type === 'List' || fieldType.type === 'Set' || fieldType.type === 'Array') &&
        'inner' in fieldType &&
        typeof fieldType.inner === 'object' &&
        'type' in fieldType.inner &&
        (fieldType.inner as any).type === 'Custom' &&
        'type_id' in fieldType.inner &&
        (fieldType.inner as any).type_id === typeId;

      if (isArrayMatch && Array.isArray(updatedRecord[fieldName])) {
        // Filter out the struct reference
        updatedRecord[fieldName] = updatedRecord[fieldName].filter(
          (item: any) => !(item && typeof item === 'object' && '$key' in item && item.$key === displayValue)
        );
      }
    });

    return updatedRecord;
  });

  return {
    ...outputConfig,
    output: updatedOutput
  };
}
