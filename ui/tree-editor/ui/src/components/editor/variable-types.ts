// Variable system types for reusable variables in the editor

export type VariableType = 'string' | 'numeric' | 'list' | 'list_string' | 'list_numeric';

export interface BaseVariable {
  type: VariableType;
}

export interface StringVariable extends BaseVariable {
  type: 'string';
  value: string;
}

export interface NumericVariable extends BaseVariable {
  type: 'numeric';
  value: number;
}

export type Variable = StringVariable | NumericVariable;

// Variable with name for storage (ID is the dictionary key)
export interface VariableInfo {
  name: string;    // Display name (can be renamed)
  var_type: VariableType;  // Match backend field name
  value: string | number;
}

export type VariableMap = Record<string, VariableInfo>; // Keyed by ID

// Color system for variables
export const VARIABLE_COLORS = [
  '#3B82F6', // blue
  '#10B981', // emerald
  '#8B5CF6', // violet
  '#F59E0B', // amber
  '#EF4444', // red
  '#06B6D4', // cyan
  '#84CC16', // lime
  '#F97316', // orange
  '#EC4899', // pink
  '#6366F1', // indigo
];

// Simple hash function to convert string to number
const hashString = (str: string): number => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return Math.abs(hash);
};

// Get consistent color for a variable ID
export const getVariableColorById = (variableId: string): string => {
  const hash = hashString(variableId);
  const colorIndex = hash % VARIABLE_COLORS.length;
  return VARIABLE_COLORS[colorIndex];
};

// Legacy function - kept for backward compatibility
export const getVariableColor = (index: number): string => {
  return VARIABLE_COLORS[index % VARIABLE_COLORS.length];
};

// Type guards for variable types  
export const isStringVariable = (variable: Variable | VariableInfo): boolean => {
  return ('type' in variable ? variable.type : variable.var_type) === 'string';
};

export const isNumericVariable = (variable: Variable | VariableInfo): boolean => {
  return ('type' in variable ? variable.type : variable.var_type) === 'numeric';
};

// Default values for each variable type
export const getDefaultValueForType = (type: VariableType): string | number => {
  switch (type) {
    case 'string':
      return '';
    case 'numeric':
      return 0;
    default:
      throw new Error(`Unknown variable type: ${type}`);
  }
};

// ID generation for variables
export const generateVariableId = (): string => {
  return `var_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
};

// Utility functions for ID-based lookups
export const getVariableById = (id: string, variables: VariableMap): VariableInfo | null => {
  return variables[id] || null;
};

export const getVariableByName = (name: string, variables: VariableMap): VariableInfo | null => {
  return Object.values(variables).find(v => v.name === name) || null;
};

export const getAllVariableIds = (variables: VariableMap): string[] => {
  return Object.keys(variables);
};

export const getAllVariableNames = (variables: VariableMap): string[] => {
  return Object.values(variables).map(v => v.name);
};

// Validation functions
export const validateVariableName = (name: string, existingVariables: VariableMap, excludeId?: string): string | null => {
  if (!name.trim()) {
    return 'Variable name cannot be empty';
  }

  const trimmedName = name.trim();

  // Check for Python variable naming: lowercase letters, numbers, underscores, must start with letter
  if (!/^[a-z][a-z0-9_]*$/.test(trimmedName)) {
    return 'Variable name must be lowercase and contain only letters, numbers, and underscores, starting with a letter';
  }

  // Check for name conflicts (excluding the variable being edited by ID)
  const existingVariable = getVariableByName(trimmedName, existingVariables);
  if (existingVariable) {
    // Find the ID of the existing variable to compare with excludeId
    const existingVariableId = Object.keys(existingVariables).find(id => existingVariables[id] === existingVariable);
    if (existingVariableId !== excludeId) {
      return 'Variable name already exists';
    }
  }

  return null;
};

export const validateVariableValue = (type: VariableType, value: any): string | null => {
  switch (type) {
    case 'string':
      if (typeof value !== 'string') {
        return 'Value must be a string';
      }
      break;
    case 'numeric':
      if (typeof value !== 'number' || isNaN(value)) {
        return 'Value must be a valid number';
      }
      break;
  }
  return null;
};

// Variable reference pattern for ${} syntax (not $ as you mentioned that causes conflicts)
export const VARIABLE_REFERENCE_PATTERN = /\$\{([a-zA-Z_][a-zA-Z0-9_-]*)\}/g;

// Extract variable references from text
export const extractVariableReferences = (text: string): string[] => {
  const matches = text.match(VARIABLE_REFERENCE_PATTERN);
  if (!matches) return [];
  
  return matches.map(match => {
    const nameMatch = match.match(/\$\{([a-zA-Z_][a-zA-Z0-9_-]*)\}/);
    return nameMatch ? nameMatch[1] : '';
  }).filter(name => name.length > 0);
};

// Replace variable references with actual values
export const replaceVariableReferences = (text: string, variables: VariableMap): string => {
  return text.replace(VARIABLE_REFERENCE_PATTERN, (match, varName) => {
    const variable = variables[varName];
    if (!variable) {
      return match; // Keep original if variable not found
    }
    
    // Convert value to string for replacement
    return String(variable.value);
  });
};

// Render normalized variables - substitute variables and normalize values (ID-based)
export const renderNormalizedVariables = <T>(
  variableIds: string[],
  directValues: T[],
  variableMap: VariableMap,
  normalizer: (values: T[]) => T[]
): T[] => {
  // Substitute variables with their actual values
  const substitutedValues: T[] = [];
  
  // Add direct values first
  substitutedValues.push(...directValues);
  
  // Add values from variables (using IDs)
  for (const varId of variableIds) {
    const variable = getVariableById(varId, variableMap);
    if (variable) {
      substitutedValues.push(variable.value as T);
    }
  }
  
  // Normalize the combined values
  return normalizer(substitutedValues);
};

// Normalizer functions
export const normalizeCategoricalValues = (values: number[]): number[] => {
  const uniqueValues = new Set(values.filter(v => typeof v === 'number' && !isNaN(v)));
  return Array.from(uniqueValues).sort((a, b) => a - b);
};

export const normalizeThresholdValues = (values: number[]): number[] => {
  const uniqueValues = new Set(values.filter(v => typeof v === 'number' && !isNaN(v)));
  return Array.from(uniqueValues).sort((a, b) => a - b);
};

export const normalizeStringValues = (values: string[]): string[] => {
  const seen = new Set<string>();
  const result: string[] = [];
  
  for (const value of values) {
    // Accept all strings (including empty strings and whitespace-only strings)
    // Only filter out non-strings and duplicates
    if (typeof value === 'string' && !seen.has(value)) {
      seen.add(value);
      result.push(value);
    }
  }
  
  return result;
};