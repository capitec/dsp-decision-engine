import { Node } from './types';

/**
 * Helper to extract feature name from v3 Feature type (string | ComputedFeature)
 */
function getFeatureName(feature: any): string | null {
  if (typeof feature === 'string') return feature;
  if (feature && typeof feature === 'object' && 'expression' in feature) {
    return feature.expression;
  }
  return null;
}

/**
 * `schema` with every column the nodes read that it lacks appended, typed as decikit reads it: a string match makes
 * a String, an is_true/is_false a Boolean, anything else a Float64. A computed feature is not a column.
 */
export function withNodeFeatures(schema: [string, any][], nodes: Node[]): [string, any][] {
  const known = new Set(schema.map(([name]) => name));
  const found = new Map<string, string>();
  const read = (feature: unknown, op?: string) => {
    if (typeof feature !== 'string' || !feature || known.has(feature)) return;
    const dtype = op === 'string_match' ? 'String' : op === 'is_true' || op === 'is_false' ? 'Boolean' : 'Float64';
    if (!found.has(feature) || dtype !== 'Float64') found.set(feature, dtype);
  };
  const readCondition = (c: any) => (c?.conditions ? c.conditions.forEach(readCondition) : read(c?.feature, c?.op));
  for (const node of nodes) {
    const data: any = node.data;
    if (data.type === 'unary') read(data.condition?.feature, data.condition?.op);
    else if (data.type === 'cases') read(data.feature, data.op);
    else if (data.type === 'composite') data.conditions?.forEach(readCondition);
  }
  return found.size ? [...schema, ...found] : schema;
}

/**
 * Rename a feature across all nodes (v3).
 */
export function renameFeature(
  nodes: Node[],
  oldName: string,
  newName: string
): Node[] {
  return nodes.map(node => {
    if ('feature' in node.data) {
      const featureName = getFeatureName(node.data.feature);
      if (featureName === oldName) {
        return {
          ...node,
          data: {
            ...node.data,
            feature: newName,
          },
        };
      }
    }
    // For unary nodes, check condition.feature
    if (node.data.type === 'unary' && 'condition' in node.data) {
      const featureName = getFeatureName(node.data.condition.feature);
      if (featureName === oldName) {
        return {
          ...node,
          data: {
            ...node.data,
            condition: {
              ...node.data.condition,
              feature: newName,
            },
          },
        };
      }
    }
    return node;
  });
}

/**
 * Get all nodes that use a specific feature (v3).
 */
export function getNodesUsingFeature(
  nodes: Node[],
  featureName: string
): Node[] {
  return nodes.filter(node => {
    if ('feature' in node.data) {
      const name = getFeatureName(node.data.feature);
      if (name === featureName) return true;
    }
    // For unary nodes, check condition.feature
    if (node.data.type === 'unary' && 'condition' in node.data) {
      const name = getFeatureName(node.data.condition.feature);
      if (name === featureName) return true;
    }
    return false;
  });
}

/**
 * Validate a feature name.
 * Returns error message if invalid, null if valid.
 */
export function validateFeatureName(name: string): string | null {
  if (!name.trim()) {
    return 'Feature name cannot be empty';
  }
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
    return 'Must be a valid identifier (letters, numbers, underscores, cannot start with number)';
  }
  return null;
}
