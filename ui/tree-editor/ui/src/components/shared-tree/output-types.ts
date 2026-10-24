// Output configuration types that match the backend PED structure exactly
// Based on the flat_rules.py example structure

import { PolarsType, PolarsSchema } from '@/types/datasource-types';

// Shared v2 types - canonical definitions in tree-types.ts
import type {
  UITypeDef as _UITypeDef,
  UITypeDefStruct as _UITypeDefStruct,
  UITypeDefCategorical as _UITypeDefCategorical,
  UITypeDefs as _UITypeDefs,
} from '@/types/tree-types';

// Re-export shared types so existing consumers of './output-types' don't break
export type UITypeDef = _UITypeDef;
export type UITypeDefStruct = _UITypeDefStruct;
export type UITypeDefCategorical = _UITypeDefCategorical;
export type UITypeDefs = _UITypeDefs;

// Alias for compatibility
export type PolarsDataType = PolarsType;

// Schema structure - array of [fieldName, fieldType] tuples to maintain order
// Example: [["Action", {"type": "Struct", "fields": [["id", "Int64"], ["Type", "String"]]}], ["Outcome", "Categorical"], ["number", "Float64"]]
export type OutputSchema = PolarsSchema;

// UI Schema structure - simplified array of [fieldName, fieldConfig] tuples
// Example: [["Action", {"type": "struct", "type_id": "uuid"}], ["Outcome", {"type": "categorical", "type_id": "uuid2"}]]
export type UISchema = Array<[string, UISchemaField]>;

export type UISchemaField =
  | UISchemaFieldStruct
  | UISchemaFieldCategorical
  | UISchemaFieldPrimitive;

export interface UISchemaFieldStruct {
  type: 'struct';
  type_id: string; // References a UI type definition
}

export interface UISchemaFieldCategorical {
  type: 'categorical';
  type_id: string; // References a UI type definition
}

export interface UISchemaFieldPrimitive {
  type: 'primitive';
  polars_type: string; // Direct polars type like 'String', 'Int64', etc.
}

// Output data array - actual records that can be selected in leaf nodes
// Example: [{"Action": {"id": 1, "Type": "NCALimitAmount"}, "number": 10, "Outcome": "Action1"}]
export type OutputRecord = Record<string, any>;

// Complete output configuration matching backend structure (NEW SIMPLIFIED FORMAT)
export interface OutputConfiguration {
  dtypes: OutputSchema;          // Merged schema with type references via type_id in extra fields
  type_defs: UITypeDefs;         // Reusable custom type definitions
  output: OutputRecord[];        // Actual data records
  default: Record<string, any>;  // Default values
  display_fmt?: string;          // Display format template for output labels (e.g., "${Action}: ${Description}")
}

// Validation helpers
export const validateOutputSchema = (schema: OutputSchema | Record<string, any>): string[] => {
  const errors: string[] = [];
  
  // Handle both old object format and new array format
  let schemaArray: OutputSchema;
  if (Array.isArray(schema)) {
    schemaArray = schema;
  } else {
    // Convert old object format to array format
    schemaArray = Object.entries(schema);
  }
  
  if (schemaArray.length === 0) {
    errors.push('At least one output field is required');
  }
  
  const fieldNames = schemaArray.map(([name]) => name);
  const uniqueNames = new Set(fieldNames);
  
  if (uniqueNames.size !== fieldNames.length) {
    errors.push('Field names must be unique');
  }
  
  // Validate each field name
  for (const [fieldName] of schemaArray) {
    if (!fieldName.trim()) {
      errors.push('Field names cannot be empty');
    }
    
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(fieldName)) {
      errors.push(`Field name "${fieldName}" is invalid. Use letters, numbers, and underscores only.`);
    }
  }
  
  return errors;
};

export const validateUITypeDef = (typeDef: UITypeDef): string[] => {
  const errors: string[] = [];

  if (!typeDef.name.trim()) {
    errors.push('Type name is required');
  }

  if (typeDef.type === 'struct') {
    if (typeDef.definition.fields.length === 0) {
      errors.push('Struct type must have at least one field');
    }

    if (typeDef.definition.data.length === 0) {
      errors.push('Struct type must have at least one data record');
    }

    // Validate display field (required)
    if (!typeDef.definition.display_field || typeDef.definition.display_field.trim() === '') {
      errors.push('Struct type must have a display field selected');
    } else {
      const fieldExists = typeDef.definition.fields.find(([name]) => name === typeDef.definition.display_field);
      if (!fieldExists) {
        errors.push('Display field must be one of the defined fields');
      } else {
        // Check uniqueness of display field values
        const displayFieldName = typeDef.definition.display_field;
        const displayValues = typeDef.definition.data.map(record => record[displayFieldName]);
        const uniqueValues = new Set(displayValues.filter(v => v !== undefined && v !== '' && v !== null));
        const nonEmptyValues = displayValues.filter(v => v !== undefined && v !== '' && v !== null);

        if (uniqueValues.size !== nonEmptyValues.length) {
          errors.push('Display field values must be unique across all records');
        }

        // Check that display field values are not empty
        if (displayValues.some(v => v === undefined || v === '' || v === null)) {
          errors.push('Display field values cannot be empty');
        }
      }
    }
  } else if (typeDef.type === 'categorical') {
    if (typeDef.definition.categories.length === 0) {
      errors.push('Categorical type must have at least one category');
    }
  }

  return errors;
};

// Helper functions to create default values
export const createDefaultOutputConfiguration = (): OutputConfiguration => ({
  dtypes: [],
  type_defs: {},
  output: [],
  default: {}
});

export const createDefaultStructTypeDef = (): UITypeDefStruct => ({
  name: '',
  type: 'struct',
  definition: {
    fields: [['id', 'String']], // Always start with a key field
    data: [],
    display_field: 'id' // Set the first field as display field by default
  }
});

export const createDefaultCategoricalTypeDef = (): UITypeDefCategorical => ({
  name: '',
  type: 'categorical', 
  definition: {
    categories: []
  }
});

// Helper functions to work with dtypes (unified schema format)
export interface DTypeEntry {
  fieldName: string;
  fieldType: PolarsDataType;
  typeId?: string; // For custom types
  typeKind?: 'struct' | 'categorical'; // For custom types
}

// Extract structured information from dtypes entry
export function parseDTypeEntry(fieldName: string, fieldType: PolarsDataType, type_defs?: UITypeDefs): DTypeEntry {
  const entry: DTypeEntry = { fieldName, fieldType };

  // Check if this is a Custom type reference
  if (typeof fieldType === 'object' && fieldType !== null && 'type' in fieldType) {
    const explicitType = fieldType as any;

    if (explicitType.type === 'Custom' && explicitType.type_id) {
      entry.typeId = explicitType.type_id;

      // Determine type kind from type_defs
      if (type_defs && explicitType.type_id in type_defs) {
        entry.typeKind = type_defs[explicitType.type_id].type;
      }
    }
  }

  return entry;
}

// Create a dtypes entry for a field
export function createDTypeEntry(
  fieldName: string,
  typeSelection: string,
  customTypeId?: string,
  type_defs?: UITypeDefs
): [string, PolarsDataType] {
  if ((typeSelection === 'custom-struct' || typeSelection === 'custom-categorical') && customTypeId && type_defs) {
    // Custom type - create Custom type with type_id
    return [fieldName, {
      type: 'Custom',
      type_id: customTypeId
    } as any];
  } else {
    // Primitive type - just use the string
    return [fieldName, typeSelection as PolarsDataType];
  }
}

// Update dtypes when a custom type definition changes
export function updateDTypesForTypeDefChange(
  dtypes: OutputSchema,
  typeId: string,
  updatedTypeDef: UITypeDef
): OutputSchema {
  return dtypes.map(([fieldName, fieldType]) => {
    // Check if this field uses the updated type
    if (typeof fieldType === 'object' && fieldType !== null && 'type' in fieldType) {
      const explicitType = fieldType as any;
      if (explicitType.type === 'Custom' && explicitType.type_id === typeId) {
        // Keep it as a Custom type reference - just return as-is
        // The type definition is stored separately in type_defs
        return [fieldName, {
          type: 'Custom',
          type_id: typeId
        } as any];
      }
    }
    return [fieldName, fieldType];
  });
}

// All polars primitive types for dropdown selection
// Organized by category for better usability
export const COMMON_POLARS_TYPES: string[] = [
  // Most common types (top of dropdown)
  'String',
  'Int64',
  'Float64',
  'Boolean',
  'Date',
  'Datetime',

  // Other numeric types
  'Int8',
  'Int16',
  'Int32',
  'Int128',
  'UInt8',
  'UInt16',
  'UInt32',
  'UInt64',
  'UInt128',
  'Float16',
  'Float32',

  // Binary and null
  'Binary',
  'Null',

  // Temporal types
  'Time',
  'Duration',

  // Complex types (displayed but may need special handling)
  'Decimal',
  'Categorical',
  'List',
  'Set',
  'Array',
];