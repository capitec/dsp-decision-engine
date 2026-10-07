import type { UITypeDefs, OutputConfiguration } from './output-types';

/**
 * Updates all references to struct records when display field values change.
 *
 * Struct references are stored as {"$key": "display_value"} where display_value
 * is the value of the display field. This function applies a complete mapping at once
 * to avoid sequential update issues (e.g., a→b, b→a would fail if done sequentially).
 *
 * @param mapping - Map of old display values to new display values
 * @param uiTypeDefs - All custom type definitions
 * @param outputConfig - The output configuration (optional)
 * @returns Updated type definitions and output configuration
 */
export function updateStructReferencesWithMapping(
  mapping: Record<string, string>,
  uiTypeDefs: UITypeDefs,
  outputConfig?: OutputConfiguration
): { updatedTypeDefs: UITypeDefs; updatedOutputConfig?: OutputConfiguration } {
  const updatedTypeDefs = { ...uiTypeDefs };
  let updatedOutputConfig = outputConfig ? { ...outputConfig } : undefined;

  // Helper to update a struct reference if it matches
  const updateStructRef = (value: any): any => {
    if (typeof value === 'object' && value !== null && '$key' in value) {
      const oldKey = value.$key;
      if (oldKey in mapping) {
        return { $key: mapping[oldKey] };
      }
    }
    return value;
  };

  // Helper to recursively update any value (handles nested structures)
  const updateValue = (value: any): any => {
    if (value === null || value === undefined) return value;

    if (Array.isArray(value)) {
      return value.map(updateValue);
    }

    if (typeof value === 'object') {
      // Check if this is a struct reference
      const updated = updateStructRef(value);
      if (updated !== value) return updated;

      // Otherwise recursively update nested objects
      const newObj: any = {};
      for (const [key, val] of Object.entries(value)) {
        newObj[key] = updateValue(val);
      }
      return newObj;
    }

    return value;
  };

  // Update output configuration if provided
  if (updatedOutputConfig) {
    // Update all output data
    updatedOutputConfig.output = updatedOutputConfig.output.map(record => {
      const newRecord: any = {};
      for (const [key, value] of Object.entries(record)) {
        newRecord[key] = updateValue(value);
      }
      return newRecord;
    });

    // Update default
    if (updatedOutputConfig.default) {
      const newDefault: any = {};
      for (const [key, value] of Object.entries(updatedOutputConfig.default)) {
        newDefault[key] = updateValue(value);
      }
      updatedOutputConfig.default = newDefault;
    }
  }

  // Update other custom types that might use this struct
  for (const [otherTypeId, otherTypeDef] of Object.entries(updatedTypeDefs)) {
    if (otherTypeDef.type === 'struct') {
      const structDef = otherTypeDef;
      const updatedData = structDef.definition.data.map(record => {
        const newRecord: any = {};
        for (const [key, value] of Object.entries(record)) {
          newRecord[key] = updateValue(value);
        }
        return newRecord;
      });

      updatedTypeDefs[otherTypeId] = {
        ...structDef,
        definition: {
          ...structDef.definition,
          data: updatedData
        }
      };
    }
  }

  return { updatedTypeDefs, updatedOutputConfig };
}

/**
 * Legacy single-value update function. Use updateStructReferencesWithMapping for multiple changes.
 * @deprecated Use updateStructReferencesWithMapping with a mapping object instead
 */
export function updateStructReferences(
  typeId: string,
  oldDisplayValue: any,
  newDisplayValue: any,
  displayFieldName: string,
  uiTypeDefs: UITypeDefs,
  outputConfig?: OutputConfiguration
): { updatedTypeDefs: UITypeDefs; updatedOutputConfig?: OutputConfiguration } {
  return updateStructReferencesWithMapping(
    { [oldDisplayValue]: newDisplayValue },
    uiTypeDefs,
    outputConfig
  );
}
