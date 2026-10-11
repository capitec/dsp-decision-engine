/**
 * Utility to normalize dtypes and type_defs format from backend to consistent array format.
 *
 * When data is stored in JSONB and retrieved from the database, these can come back as:
 * - Array format (correct): [["Action", {...}], ["Score", {...}]]
 * - Object format (needs fixing): {"Action": {...}, "Score": {...}}
 *
 * This utility ensures the frontend always receives the array format.
 */

import type { TreeOutput, PolarsSchema, UITypeDefs, UITypeDef } from '@/types/tree-types';

/**
 * Normalize dtypes to array-of-tuples format.
 * Handles both array and object formats from the backend.
 */
export function normalizeDtypes(dtypes: any): PolarsSchema {
  // If it's already an array, return as-is
  if (Array.isArray(dtypes)) {
    return dtypes as PolarsSchema;
  }

  // If it's an object/dict, convert to array of tuples
  if (typeof dtypes === 'object' && dtypes !== null) {
    return Object.entries(dtypes) as PolarsSchema;
  }

  // Fallback: empty array
  console.warn('Unexpected dtypes format:', dtypes);
  return [];
}

/**
 * Normalize a single UITypeDef to ensure fields is in array format (for struct types).
 */
export function normalizeTypeDef(typeDef: UITypeDef): UITypeDef {
  if (typeDef.type === 'struct') {
    const fields = typeDef.definition.fields;

    // If fields is an object, convert to array of tuples
    if (!Array.isArray(fields) && typeof fields === 'object' && fields !== null) {
      return {
        ...typeDef,
        definition: {
          ...typeDef.definition,
          fields: Object.entries(fields) as Array<[string, any]>
        }
      };
    }
  }

  // Return as-is for categorical types or if already in correct format
  return typeDef;
}

/**
 * Normalize type_defs to ensure all struct fields are in array format.
 */
export function normalizeTypeDefs(typeDefs: UITypeDefs): UITypeDefs {
  const normalized: UITypeDefs = {};

  for (const [key, typeDef] of Object.entries(typeDefs)) {
    normalized[key] = normalizeTypeDef(typeDef);
  }

  return normalized;
}

/**
 * Normalize TreeOutput to ensure dtypes and type_defs are in array format.
 * This should be called whenever TreeOutput comes from the backend.
 */
export function normalizeTreeOutput(output: TreeOutput): TreeOutput {
  return {
    ...output,
    dtypes: normalizeDtypes(output.dtypes),
    type_defs: normalizeTypeDefs(output.type_defs || {}),
  };
}
