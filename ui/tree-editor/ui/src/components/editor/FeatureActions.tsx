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
 * Extract unique feature names from all nodes in the tree (v3).
 */
export function extractFeatures(nodes: Node[]): string[] {
  const features = new Set<string>();
  for (const node of nodes) {
    if ('feature' in node.data) {
      const featureName = getFeatureName(node.data.feature);
      if (featureName) {
        features.add(featureName);
      }
    }
    // For unary nodes, extract from condition.feature
    if (node.data.type === 'unary' && 'condition' in node.data) {
      const featureName = getFeatureName(node.data.condition.feature);
      if (featureName) {
        features.add(featureName);
      }
    }
  }
  return Array.from(features).sort();
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
