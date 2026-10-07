/**
 * Utility to parse Pydantic validation errors and identify affected nodes/features/outputs
 */

export interface ValidationError {
  loc: (string | number)[];
  msg: string;
  type: string;
  input?: any;
}

export interface ParsedValidationError {
  path: string;
  message: string;
  type: string;
  // Identify what part of the tree is affected
  affectedNodeId?: string;
  affectedNodeIndex?: number;
  // Path inside the affected node, without decider's union tags (empty = the node itself)
  fieldPath?: (string | number)[];
  affectedFeature?: string;
  affectedOutput?: boolean;
  affectedParameters?: boolean;
  // Raw error for debugging
  raw: ValidationError;
}

const NODE_TAGS = new Set(['leaf', 'unary', 'composite', 'cases', 'cases:ranges', 'cases:isin', 'cases:string_match']);
const OP_TAGS = new Set(['<', '<=', '==', '>', '>=', '!=', 'between', 'isin', 'string_match', 'is_true', 'is_false', 'is_null', 'is_not_null']);
// Union-branch tags pydantic appends for every branch it tried: they carry no location information.
const BRANCH_TAG = /^(int|float|str|bool|ParamRef|list\[.*|function-.*|union\[.*|(IsIn|Range|StringMatch)Condition)$/;
const CONDITION_FOR: Record<string, string> = {
  'cases:isin': 'IsInCondition',
  'cases:ranges': 'RangeCondition',
  'cases:string_match': 'StringMatchCondition',
};

/**
 * Strip decider's union tags from the part of a loc below a node or rule.
 * Cuts at the first union-branch tag, so every branch tried for one field maps to the same path.
 * Returns null for errors from a cases-condition branch that does not match the node's own op.
 */
export function cleanLoc(loc: (string | number)[]): (string | number)[] | null {
  const out: (string | number)[] = [];
  let nodeTag: string | null = null;
  let afterTag = false;
  for (let i = 0; i < loc.length; i++) {
    const seg = loc[i];
    if (typeof seg === 'string') {
      if (BRANCH_TAG.test(seg)) {
        const wanted = nodeTag ? CONDITION_FOR[nodeTag] : undefined;
        if (wanted && seg.endsWith('Condition') && seg !== wanted) return null;
        break;
      }
      if (NODE_TAGS.has(seg)) {
        nodeTag = seg;
        afterTag = true;
        continue;
      }
      if (OP_TAGS.has(seg) && (afterTag || loc[i - 1] === 'condition')) {
        afterTag = false;
        continue;
      }
    }
    afterTag = false;
    out.push(seg);
  }
  return out;
}

/** Collapse errors that share a key, keeping the one with the deepest loc (the real cause). */
export function dedupeByPath<T>(
  items: Array<{ key: string; error: T; loc: (string | number)[] }>
): T[] {
  const best = new Map<string, { error: T; depth: number }>();
  for (const { key, error, loc } of items) {
    const prev = best.get(key);
    if (!prev || loc.length > prev.depth) best.set(key, { error, depth: loc.length });
  }
  return Array.from(best.values(), b => b.error);
}

/**
 * Parse validation errors from API response and identify affected parts of the tree
 *
 * @param errors - Array of validation errors from API
 * @param nodes - Optional array of nodes to map indices to IDs
 */
export function parseValidationErrors(
  errors: ValidationError[],
  nodes?: Array<{ id: string }>
): ParsedValidationError[] {
  const keyed = errors.flatMap(error => {
    const { loc, msg, type } = error;
    const path = loc.join(' → ');

    const parsed: ParsedValidationError = {
      path,
      message: msg,
      type,
      raw: error
    };

    // Parse location to identify affected parts
    // Example loc: ["v3", "nodes", 0, "data", "unary", "condition", ">", "threshold", "ParamRef", "param"]
    let key = path;

    // Build errors have an empty loc; the message may name the node ("tree node '<id>': ...")
    if (type === 'build_error') {
      const nodeId = /tree node '([^']+)'/.exec(msg)?.[1];
      if (nodeId) parsed.affectedNodeId = nodeId;
      return [{ key: `build: ${msg}`, error: parsed, loc }];
    }

    let i = 0;
    while (i < loc.length) {
      const segment = loc[i];

      // Find nodes array
      if (segment === 'nodes' && i + 1 < loc.length && typeof loc[i + 1] === 'number') {
        const nodeIndex = loc[i + 1] as number;
        parsed.affectedNodeIndex = nodeIndex;
        const field = cleanLoc(loc.slice(i + 2).filter(seg => seg !== 'data'));
        if (!field) return [];
        parsed.fieldPath = field;
        parsed.path = field.join(' → ');
        key = `node ${nodeIndex}: ${parsed.path}`;

        // Map index to node ID if nodes array provided
        if (nodes && nodeIndex >= 0 && nodeIndex < nodes.length) {
          parsed.affectedNodeId = nodes[nodeIndex].id;
        }

        i += 2;
        continue;
      }

      // Find input_schema
      if (segment === 'input_schema') {
        if (i + 1 < loc.length && typeof loc[i + 1] === 'number') {
          const featureIndex = loc[i + 1] as number;
          // We don't have feature names here, just index
          parsed.affectedFeature = `Feature at index ${featureIndex}`;
        } else {
          parsed.affectedFeature = 'Features configuration';
        }
        i += 2;
        continue;
      }

      // Find output configuration
      if (segment === 'output') {
        parsed.affectedOutput = true;
        i++;
        continue;
      }

      // Find parameters configuration
      if (segment === 'parameters') {
        parsed.affectedParameters = true;
        i++;
        continue;
      }

      i++;
    }

    return [{ key, error: parsed, loc }];
  });
  return dedupeByPath(keyed);
}

/**
 * Group validation errors by affected component
 */
export function groupValidationErrors(parsed: ParsedValidationError[]) {
  const groups = {
    nodes: [] as ParsedValidationError[],
    features: [] as ParsedValidationError[],
    output: [] as ParsedValidationError[],
    parameters: [] as ParsedValidationError[],
    subtrees: [] as ParsedValidationError[],
    other: [] as ParsedValidationError[]
  };

  for (const error of parsed) {
    if (error.affectedNodeId || error.affectedNodeIndex !== undefined) {
      groups.nodes.push(error);
    } else if (error.affectedFeature) {
      groups.features.push(error);
    } else if (error.affectedOutput) {
      groups.output.push(error);
    } else if (error.affectedParameters) {
      groups.parameters.push(error);
    } else if (error.raw.loc.includes('subtrees')) {
      groups.subtrees.push(error);
    } else {
      groups.other.push(error);
    }
  }

  return groups;
}

/**
 * Format validation errors into a user-friendly message
 */
export function formatValidationErrors(parsed: ParsedValidationError[]): string {
  if (parsed.length === 0) return '';

  const groups = groupValidationErrors(parsed);
  const lines: string[] = [];

  if (groups.nodes.length > 0) {
    lines.push('**Node Issues:**');
    groups.nodes.forEach(err => {
      const nodeInfo = err.affectedNodeId
        ? `Node "${err.affectedNodeId}"`
        : `Node at index ${err.affectedNodeIndex}`;
      const field = err.fieldPath?.length ? ` (${err.fieldPath.join(' → ')})` : '';
      lines.push(`• ${nodeInfo}${field}: ${err.message}`);
    });
  }

  if (groups.features.length > 0) {
    lines.push('', '**Feature Issues:**');
    groups.features.forEach(err => {
      lines.push(`• ${err.affectedFeature}: ${err.message}`);
    });
  }

  if (groups.output.length > 0) {
    lines.push('', '**Output Configuration Issues:**');
    groups.output.forEach(err => {
      lines.push(`• ${err.message}`);
    });
  }

  if (groups.parameters.length > 0) {
    lines.push('', '**Parameter Issues:**');
    groups.parameters.forEach(err => {
      lines.push(`• ${err.message}`);
    });
  }

  if (groups.subtrees.length > 0) {
    lines.push('', '**Subtree Issues:**');
    groups.subtrees.forEach(err => {
      lines.push(`• ${err.message}`);
    });
  }

  if (groups.other.length > 0) {
    lines.push('', '**Other Issues:**');
    groups.other.forEach(err => {
      lines.push(`• ${err.path ? `${err.path}: ` : ''}${err.message}`);
    });
  }

  return lines.join('\n');
}

/**
 * Get set of node IDs that have validation errors
 */
export function getInvalidNodeIds(parsed: ParsedValidationError[]): Set<string> {
  const ids = new Set<string>();
  for (const error of parsed) {
    if (error.affectedNodeId) {
      ids.add(error.affectedNodeId);
    }
  }
  return ids;
}

/**
 * Get set of node indices that have validation errors
 */
export function getInvalidNodeIndices(parsed: ParsedValidationError[]): Set<number> {
  const indices = new Set<number>();
  for (const error of parsed) {
    if (error.affectedNodeIndex !== undefined) {
      indices.add(error.affectedNodeIndex);
    }
  }
  return indices;
}

/**
 * Get section error flags for highlighting sidebar sections
 */
export function getSectionErrors(parsed: ParsedValidationError[]) {
  const groups = groupValidationErrors(parsed);
  return {
    hasFeatureErrors: groups.features.length > 0,
    hasOutputErrors: groups.output.length > 0,
    hasParameterErrors: groups.parameters.length > 0,
    hasSubtreeErrors: groups.subtrees.length > 0,
  };
}
