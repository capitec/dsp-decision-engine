import { renderNormalizedVariables, normalizeCategoricalValues, normalizeThresholdValues, normalizeStringValues, VariableMap } from '../components/editor/variable-types';
import { isParameterReference } from '@/types/tree-types';
import { paramRefName } from '@/utils/param-refs';

export const OP_SYMBOL_MAP: Record<string, string> = {
  "<=": "≤",
  "<": "<",
  "==": "=",
  ">": ">",
  ">=": "≥",
};

export const DEFAULT_SYMBOL = "∅";

export interface EdgeOption {
  value: number;
  label: string;
  description: string;
}

// Helper to format a value that might be a ParameterReference
const formatValue = (value: any): string => {
  if (isParameterReference(value)) {
    return `$${paramRefName(value)}`;
  }
  return String(value);
};

// Private functions for each node type
const getNumericalTestOptions = (nodeData: any, variables: VariableMap): EdgeOption[] => {
  const op = nodeData.comparison_op || "<=";
  const threshold = nodeData.threshold || 0;
  const symbol = OP_SYMBOL_MAP[op] || op;
  const thresholdDisplay = formatValue(threshold);

  return [
    { value: 0, label: `${symbol} ${thresholdDisplay}`, description: "Left branch (condition true)" },
    { value: 1, label: "otherwise", description: "Right branch (condition false)" }
  ];
};

const getCategoricalTestOptions = (nodeData: any, variables: VariableMap): EdgeOption[] => {
  const categoryList = nodeData.category_list || [];

  // v2: category_list can contain numbers or ParameterReference objects
  const formattedCategories = categoryList.map((cat: any) => formatValue(cat));
  const categories = `{${formattedCategories.join(', ')}}`;
  const isRightChild = nodeData.category_list_right_child === true;

  if (isRightChild) {
    // When category_list_right_child is true, branches are swapped
    return [
      { value: 0, label: "otherwise", description: "Left branch (not in categories)" },
      { value: 1, label: categories, description: "Right branch (in categories)" }
    ];
  } else {
    // Default behavior - categories on left branch
    return [
      { value: 0, label: categories, description: "Left branch (in categories)" },
      { value: 1, label: "otherwise", description: "Right branch (not in categories)" }
    ];
  }
};

const getNumericalRangeOptions = (nodeData: any, variables: VariableMap): EdgeOption[] => {
  const thresholds = nodeData.thresholds || [];
  if (thresholds.length === 0) return [];

  // v2: thresholds can contain numbers or ParameterReference objects
  // For display purposes, we show them as-is (don't try to sort ParameterReferences)
  const hasVariableRefs = thresholds.some((t: any) => isParameterReference(t));

  if (hasVariableRefs) {
    // If any threshold is a ParameterReference, show simplified labels
    const options: EdgeOption[] = [];
    for (let i = 0; i < thresholds.length + 1; i++) {
      if (i === 0) {
        options.push({
          value: 0,
          label: `< ${formatValue(thresholds[0])}`,
          description: `First branch`
        });
      } else if (i === thresholds.length) {
        options.push({
          value: i,
          label: `≥ ${formatValue(thresholds[i - 1])}`,
          description: `Last branch`
        });
      } else {
        options.push({
          value: i,
          label: `[${i}]`,
          description: `Branch ${i}`
        });
      }
    }
    return options;
  }

  // All numeric thresholds - can sort and show ranges
  const sortedThresholds = [...thresholds].sort((a, b) => a - b);
  const options: EdgeOption[] = [];

  // First range: < first threshold
  options.push({
    value: 0,
    label: `< ${sortedThresholds[0]}`,
    description: `Values less than ${sortedThresholds[0]}`
  });

  // Middle ranges: >= threshold[i] & < threshold[i+1]
  for (let i = 0; i < sortedThresholds.length - 1; i++) {
    options.push({
      value: i + 1,
      label: `≥ ${sortedThresholds[i]} & < ${sortedThresholds[i + 1]}`,
      description: `Values between ${sortedThresholds[i]} and ${sortedThresholds[i + 1]}`
    });
  }

  // Last range: >= last threshold
  const lastThreshold = sortedThresholds[sortedThresholds.length - 1];
  options.push({
    value: sortedThresholds.length,
    label: `≥ ${lastThreshold}`,
    description: `Values greater than or equal to ${lastThreshold}`
  });

  return options;
};

const getStringMatchOptions = (nodeData: any, variables: VariableMap): EdgeOption[] => {
  const nodeVariables = nodeData.variables || [];
  const directPatterns = nodeData.patterns || [];
  const resolvedPatterns = renderNormalizedVariables(nodeVariables, directPatterns, variables, normalizeStringValues);
  const matchAny = nodeData.match_any !== false;
  
  if (matchAny) {
    // Any pattern match mode - 2 branches: match or no match
    const caseIcon = nodeData.case_sensitive === false ? "i" : "";
    return [
      { value: 0, label: `patterns${caseIcon}(${resolvedPatterns.length})`, description: "Any pattern matches" },
      { value: 1, label: "no match", description: "No patterns match" }
    ];
  } else {
    // Individual pattern branches mode - one branch per pattern + no match
    const options: EdgeOption[] = resolvedPatterns.map((pattern: string, index: number) => {
      let displayLabel: string;
      let displayDescription: string;
      
      if (pattern === '') {
        // Empty string - show clearly
        displayLabel = '∅ (empty)';
        displayDescription = 'Empty string pattern matches';
      } else if (pattern.trim() === '' && pattern.length > 0) {
        // Whitespace only - analyze spaces vs tabs
        const spaceCount = (pattern.match(/ /g) || []).length;
        const tabCount = (pattern.match(/\t/g) || []).length;
        const parts: string[] = [];
        
        if (spaceCount > 0) {
          parts.push(`${spaceCount} space${spaceCount === 1 ? '' : 's'}`);
        }
        if (tabCount > 0) {
          parts.push(`${tabCount} tab${tabCount === 1 ? '' : 's'}`);
        }
        
        displayLabel = `[${parts.join(' + ')}]`;
        displayDescription = `Pattern with ${parts.join(' + ')} matches`;
      } else if (pattern.length > 20) {
        // Long patterns - truncate but preserve important parts
        const start = pattern.substring(0, 10);
        const end = pattern.substring(pattern.length - 7);
        displayLabel = `"${start}...${end}"`;
        displayDescription = `Pattern "${pattern}" matches`;
      } else {
        // Normal patterns - show as-is with quotes
        displayLabel = `"${pattern}"`;
        displayDescription = `Pattern "${pattern}" matches`;
      }
      
      return {
        value: index,
        label: displayLabel,
        description: displayDescription
      };
    });
    
    // Add "no match" option at the end
    options.push({
      value: resolvedPatterns.length,
      label: "no match",
      description: "No patterns match"
    });
    
    return options;
  }
};

const getCasesRangesOptions = (nodeData: any, variables: VariableMap): EdgeOption[] => {
  const conditions = nodeData.conditions || [];
  const endLogic = nodeData.end_logic || 'lower_inclusive';
  const strictMode = nodeData.strict_mode ?? false;

  const options: EdgeOption[] = conditions.map((cond: any, index: number) => {
    // In strict mode, infer min from previous condition's max
    let min = cond.min;
    if (strictMode && index > 0) {
      min = conditions[index - 1].max;
    }
    const max = cond.max;

    // Format range based on min/max and end_logic
    let label: string;
    let description: string;

    if (min != null && max != null) {
      // Both bounds defined
      if (endLogic === 'lower_inclusive') {
        label = `${formatValue(min)} ≤ x < ${formatValue(max)}`;
        description = `Values from ${formatValue(min)} (inclusive) to ${formatValue(max)} (exclusive)`;
      } else {
        label = `${formatValue(min)} < x ≤ ${formatValue(max)}`;
        description = `Values from ${formatValue(min)} (exclusive) to ${formatValue(max)} (inclusive)`;
      }
    } else if (min != null) {
      // Only min bound
      if (endLogic === 'lower_inclusive') {
        label = `x ≥ ${formatValue(min)}`;
        description = `Values greater than or equal to ${formatValue(min)}`;
      } else {
        label = `x > ${formatValue(min)}`;
        description = `Values greater than ${formatValue(min)}`;
      }
    } else if (max != null) {
      // Only max bound
      if (endLogic === 'lower_inclusive') {
        label = `x < ${formatValue(max)}`;
        description = `Values less than ${formatValue(max)}`;
      } else {
        label = `x ≤ ${formatValue(max)}`;
        description = `Values less than or equal to ${formatValue(max)}`;
      }
    } else {
      // No bounds (shouldn't happen but handle gracefully)
      label = `Branch ${index + 1}`;
      description = `Branch ${index + 1}`;
    }

    return {
      value: index,
      label,
      description
    };
  });

  // Add otherwise branch
  options.push({
    value: conditions.length,
    label: "otherwise",
    description: "No range conditions match"
  });

  return options;
};

const getCasesStringMatchOptions = (nodeData: any, variables: VariableMap): EdgeOption[] => {
  const conditions = nodeData.conditions || [];
  const caseIcon = nodeData.case_sensitive === false ? "i" : "";

  const options: EdgeOption[] = conditions.map((cond: any, index: number) => {
    const patterns = cond.patterns || [];

    let label: string;
    let description: string;

    if (Array.isArray(patterns)) {
      if (patterns.length === 0) {
        label = `∅`;
        description = `Empty pattern list`;
      } else if (patterns.length === 1) {
        const pattern = patterns[0];
        if (pattern === '') {
          label = `∅ (empty)${caseIcon}`;
          description = 'Empty string pattern matches';
        } else if (pattern.length > 20) {
          const start = pattern.substring(0, 10);
          const end = pattern.substring(pattern.length - 7);
          label = `"${start}...${end}"${caseIcon}`;
          description = `Pattern "${pattern}" matches`;
        } else {
          label = `"${pattern}"${caseIcon}`;
          description = `Pattern "${pattern}" matches`;
        }
      } else {
        label = `${patterns.length} patterns${caseIcon}`;
        description = `Matches any of ${patterns.length} patterns`;
      }
    } else {
      // InputRef
      label = `${formatValue(patterns)}${caseIcon}`;
      description = `Pattern from variable ${formatValue(patterns)}`;
    }

    return {
      value: index,
      label,
      description
    };
  });

  // Add otherwise branch
  options.push({
    value: conditions.length,
    label: "otherwise",
    description: "No patterns match"
  });

  return options;
};

const getCasesIsInOptions = (nodeData: any, variables: VariableMap): EdgeOption[] => {
  const conditions = nodeData.conditions || [];

  const options: EdgeOption[] = conditions.map((cond: any, index: number) => {
    const values = cond.values || [];

    let label: string;
    let description: string;

    if (Array.isArray(values)) {
      if (values.length === 0) {
        label = `{}`;
        description = `Empty value set`;
      } else if (values.length <= 3) {
        label = `{${values.map((v: any) => formatValue(v)).join(', ')}}`;
        description = `Value is in {${values.join(', ')}}`;
      } else {
        label = `{${values.slice(0, 2).map((v: any) => formatValue(v)).join(', ')}, ...${values.length - 2} more}`;
        description = `Value is in set of ${values.length} values`;
      }
    } else {
      // InputRef
      label = formatValue(values);
      description = `Values from variable ${formatValue(values)}`;
    }

    return {
      value: index,
      label,
      description
    };
  });

  // Add otherwise branch
  options.push({
    value: conditions.length,
    label: "otherwise",
    description: "Not in any value sets"
  });

  return options;
};

const getUnaryOptions = (nodeData: any, variables: VariableMap): EdgeOption[] => {
  const condition = nodeData.condition;
  if (!condition) {
    return getDefaultOptions(nodeData, variables);
  }

  const op = condition.op;
  let label: string;

  // Handle different operator types
  if (op === "<=") {
    const threshold = formatValue(condition.threshold);
    label = `≤ ${threshold}`;
  } else if (op === "<") {
    const threshold = formatValue(condition.threshold);
    label = `< ${threshold}`;
  } else if (op === "==") {
    const threshold = formatValue(condition.threshold);
    label = `= ${threshold}`;
  } else if (op === ">") {
    const threshold = formatValue(condition.threshold);
    label = `> ${threshold}`;
  } else if (op === ">=") {
    const threshold = formatValue(condition.threshold);
    label = `≥ ${threshold}`;
  } else if (op === "!=") {
    const threshold = formatValue(condition.threshold);
    label = `≠ ${threshold}`;
  } else if (op === "between") {
    const min = formatValue(condition.min);
    const max = formatValue(condition.max);
    label = `${min} ≤ x ≤ ${max}`;
  } else if (op === "isin") {
    const values = condition.values || [];
    if (Array.isArray(values) && values.length <= 3) {
      label = `∈ {${values.join(', ')}}`;
    } else if (Array.isArray(values)) {
      label = `∈ {${values.slice(0, 2).join(', ')}, ...}`;
    } else {
      label = `∈ ${formatValue(values)}`;
    }
  } else if (op === "string_match") {
    const patterns = condition.patterns || [];
    const caseIcon = condition.case_sensitive === false ? "i" : "";
    if (Array.isArray(patterns) && patterns.length === 1) {
      label = `"${patterns[0]}"${caseIcon}`;
    } else if (Array.isArray(patterns)) {
      label = `${patterns.length} patterns${caseIcon}`;
    } else {
      label = `${formatValue(patterns)}${caseIcon}`;
    }
  } else if (op === "is_true") {
    label = "is true";
  } else if (op === "is_false") {
    label = "is false";
  } else {
    // Fallback for unknown operators
    label = "then";
  }

  return [
    { value: 0, label, description: "Condition is true" },
    { value: 1, label: "otherwise", description: "Condition is false" }
  ];
};

const getDefaultOptions = (nodeData: any, variables: VariableMap): EdgeOption[] => {
  return [
    { value: 0, label: "then", description: "Left branch" },
    { value: 1, label: "otherwise", description: "Right branch" }
  ];
};

// Lookup map for node type to function
const NODE_OPTION_HANDLERS: Record<string, (nodeData: any, variables: VariableMap) => EdgeOption[]> = {
  // V2 node types
  "numerical_test_node": getNumericalTestOptions,
  "categorical_test_node": getCategoricalTestOptions,
  "numerical_range_test_node": getNumericalRangeOptions,
  "string_match_node": getStringMatchOptions,
  // V3 node types
  "cases:ranges": getCasesRangesOptions,
  "cases:string_match": getCasesStringMatchOptions,
  "cases:isin": getCasesIsInOptions,
  "unary": getUnaryOptions,
};

/**
 * Generate edge options for a node
 */
export const getNodeEdgeOptions = (nodeData: any, variables: VariableMap): EdgeOption[] => {
  // V2 uses node_type, V3 uses type
  const nodeType = nodeData?.node_type || nodeData?.type;

  // For v3 cases nodes, create a key with type:op format
  let handlerKey = nodeType;
  if (nodeType === 'cases' && nodeData?.op) {
    handlerKey = `cases:${nodeData.op}`;
  }

  const handler = NODE_OPTION_HANDLERS[handlerKey];
  return handler ? handler(nodeData, variables) : getDefaultOptions(nodeData, variables);
};

/**
 * Generate edge labels for a node (used by LabeledEdge)
 */
export const getNodeEdgeLabels = (nodeData: any, variables: VariableMap): string[] => {
  const options = getNodeEdgeOptions(nodeData, variables);
  return options.map(option => option.label);
};