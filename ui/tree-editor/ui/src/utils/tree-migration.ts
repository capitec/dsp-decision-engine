/**
 * Migration utilities for converting v1 tree data (split_feature_id: number)
 * to v2 format (feature: string).
 *
 * In normal operation the backend handles v1→v2 conversion on fetch.
 * These utilities are for local imports of v1 JSON files.
 */

/**
 * Migrate v1 nodes with split_feature_id to v2 nodes with feature names.
 *
 * @param nodes - v1 nodes with split_feature_id
 * @param features - Feature names array (indices match split_feature_id)
 * @returns Nodes with feature names instead of indices
 */
export function migrateFeatureIndicesToNames(
  nodes: any[],
  features: string[]
): any[] {
  return nodes.map(node => {
    const data = { ...node.data };

    if ('split_feature_id' in data && typeof data.split_feature_id === 'number') {
      const featureIndex = data.split_feature_id;
      if (featureIndex >= 0 && featureIndex < features.length) {
        data.feature = features[featureIndex];
      } else {
        data.feature = `unknown_feature_${featureIndex}`;
      }
      delete data.split_feature_id;
    }

    return { ...node, data };
  });
}

/**
 * Check if nodes use v1 format (split_feature_id).
 */
export function usesV1FeatureFormat(nodes: any[]): boolean {
  return nodes.some(
    node => node.data && 'split_feature_id' in node.data
  );
}

/**
 * Migrate a v1 SourceData object to v2 node format in place.
 * Handles the Record<string, NodeData> format used in SourceData.nodes.
 */
export function migrateSourceDataFeatures(
  sourceNodes: Record<string, any>,
  features: string[]
): Record<string, any> {
  const result: Record<string, any> = {};

  for (const [nodeId, nodeData] of Object.entries(sourceNodes)) {
    if ('split_feature_id' in nodeData && typeof nodeData.split_feature_id === 'number') {
      const featureIndex = nodeData.split_feature_id;
      const { split_feature_id, ...rest } = nodeData;
      result[nodeId] = {
        ...rest,
        feature: featureIndex >= 0 && featureIndex < features.length
          ? features[featureIndex]
          : `unknown_feature_${featureIndex}`,
      };
    } else {
      result[nodeId] = nodeData;
    }
  }

  return result;
}

/**
 * Migrate v1 variables (separate VariableMap with UUID keys) to v2 format
 * (inline ParameterReference objects in nodes).
 *
 * v1: Nodes have variables: string[] with IDs, VariableMap has Variable objects
 * v2: Nodes have inline ParameterReference objects
 *
 * @param nodes - Array of nodes to migrate
 * @param variableMap - v1 VariableMap object {[id]: {name, value, var_type}}
 * @returns Nodes with inline ParameterReference objects
 */
export function migrateVariablesToInline(
  nodes: any[],
  variableMap: Record<string, { name: string; value: any; var_type?: string }>
): any[] {
  if (!variableMap || Object.keys(variableMap).length === 0) {
    // No variables to migrate, just remove variables array from nodes
    return nodes.map(node => {
      const data = { ...node.data };
      if ('variables' in data) {
        delete data.variables;
      }
      return { ...node, data };
    });
  }

  return nodes.map(node => {
    const data = { ...node.data };

    // Migrate variables array to inline references
    if ('variables' in data && Array.isArray(data.variables) && data.variables.length > 0) {
      const variableId = data.variables[0]; // v1 uses single variable in array
      const variable = variableMap[variableId];

      if (variable && variable.name) {
        // Convert based on node type
        if (data.node_type === 'numerical_test_node') {
          // For numerical nodes: threshold becomes ParameterReference
          data.threshold = { name: variable.name };
        } else if (data.node_type === 'categorical_test_node') {
          // For categorical nodes: category_list becomes [ParameterReference]
          data.category_list = [{ name: variable.name }];
        } else if (data.node_type === 'numerical_range_test_node') {
          // For range nodes: thresholds becomes [ParameterReference]
          data.thresholds = [{ name: variable.name }];
        }
      }

      // Remove the old variables array
      delete data.variables;
    }

    return { ...node, data };
  });
}

/**
 * Check if nodes use v1 variable format (variables: string[]).
 */
export function usesV1VariableFormat(nodes: any[]): boolean {
  return nodes.some(
    node => node.data && 'variables' in node.data && Array.isArray(node.data.variables)
  );
}
