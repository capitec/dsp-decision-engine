import { cleanLoc, dedupeByPath } from './validation-error-parser';

/**
 * Utility to parse Pydantic validation errors for flat rules
 * Maps validation error paths to specific rules, conditions, and fields
 */

export interface FlatRuleValidationError {
  loc: (string | number)[];
  msg: string;
  type: string;
  input?: any;
}

export interface ParsedFlatRuleError {
  path: string;
  message: string;
  type: string;

  // Rule identification
  ruleIndex?: number;

  // Error location within rule
  section?: 'rule' | 'input_schema' | 'output' | 'parameters';

  // Detailed path for nested errors
  conditionPath?: (string | number)[]; // Path to the specific condition/field
  fieldName?: string; // Specific field that failed validation

  // Raw error
  raw: FlatRuleValidationError;
}

/**
 * Parse flat rule validation errors from API response
 *
 * Example error paths:
 * - ["prioritized_flat_rule", "rules", 0, "rule", "composite", "conditions", 0, "unary", "<", "threshold", "ParamRef", "param"]
 * - ["prioritized_flat_rule", "input_schema", 0]
 * - ["prioritized_flat_rule", "output", "dtypes", 0]
 *
 * Decikit's union tags are stripped from conditionPath, so it matches the editor's own rule structure
 * (["rule", "conditions", 0, "threshold"]).
 */
export function parseFlatRuleValidationErrors(
  errors: FlatRuleValidationError[]
): ParsedFlatRuleError[] {
  const keyed = errors.flatMap(error => {
    const { loc, msg, type } = error;
    const path = loc.join(' → ');

    const parsed: ParsedFlatRuleError = {
      path,
      message: msg,
      type,
      raw: error
    };

    let key = path;
    let i = 0;
    while (i < loc.length) {
      const segment = loc[i];

      // Find rule index: ["body", "project_data", "flat_rule", "rules", 0, ...]
      if (segment === 'rules' && i + 1 < loc.length && typeof loc[i + 1] === 'number') {
        parsed.ruleIndex = loc[i + 1] as number;
        parsed.section = 'rule';

        const cleaned = cleanLoc(loc.slice(i + 2));
        if (!cleaned) return [];
        parsed.conditionPath = cleaned;
        parsed.path = cleaned.join(' → ');
        key = `rule ${parsed.ruleIndex}: ${parsed.path}`;
        if (cleaned.length > 0) {

          // Extract field name (last string segment in path)
          const pathSegments = parsed.conditionPath.filter(s => typeof s === 'string');
          if (pathSegments.length > 0) {
            parsed.fieldName = pathSegments[pathSegments.length - 1] as string;
          }
        }

        break;
      }

      // Find input_schema errors
      if (segment === 'input_schema') {
        parsed.section = 'input_schema';
        if (i + 1 < loc.length && typeof loc[i + 1] === 'number') {
          parsed.conditionPath = [loc[i + 1]];
        }
        break;
      }

      // Find output errors
      if (segment === 'output') {
        parsed.section = 'output';
        parsed.conditionPath = loc.slice(i + 1);

        // Extract more detail from the path for better error messages
        const remainingPath = loc.slice(i + 1);
        if (remainingPath.length > 0) {
          const firstSegment = remainingPath[0];
          if (firstSegment === 'dtypes') {
            parsed.fieldName = 'Output Schema Types';
          } else if (firstSegment === 'data') {
            parsed.fieldName = 'Output Data';
          } else if (firstSegment === 'default') {
            parsed.fieldName = 'Default Output';
          } else if (firstSegment === 'type_defs') {
            parsed.fieldName = 'Type Definitions';
          }
        }
        break;
      }

      // Find parameter errors
      if (segment === 'parameters') {
        parsed.section = 'parameters';
        parsed.conditionPath = loc.slice(i + 1);
        break;
      }

      i++;
    }

    return [{ key, error: parsed, loc }];
  });
  return dedupeByPath(keyed);
}

/**
 * Group errors by rule index
 */
export function groupErrorsByRule(parsed: ParsedFlatRuleError[]): Map<number, ParsedFlatRuleError[]> {
  const groups = new Map<number, ParsedFlatRuleError[]>();

  for (const error of parsed) {
    if (error.ruleIndex !== undefined) {
      if (!groups.has(error.ruleIndex)) {
        groups.set(error.ruleIndex, []);
      }
      groups.get(error.ruleIndex)!.push(error);
    }
  }

  return groups;
}

/**
 * Check if a specific condition path has errors
 * Returns true if the conditionPath matches or is a parent of the error path
 */
export function hasErrorAtPath(
  errors: ParsedFlatRuleError[],
  conditionPath: (string | number)[]
): boolean {
  return errors.some(error => {
    if (!error.conditionPath) return false;

    // Check if conditionPath is a prefix of error.conditionPath
    if (conditionPath.length > error.conditionPath.length) return false;

    return conditionPath.every((segment, idx) => {
      return segment === error.conditionPath![idx];
    });
  });
}

/**
 * Get error message for a specific field path
 */
export function getErrorAtPath(
  errors: ParsedFlatRuleError[],
  conditionPath: (string | number)[]
): string | null {
  const error = errors.find(error => {
    if (!error.conditionPath) return false;

    // Exact match
    if (error.conditionPath.length !== conditionPath.length) return false;

    return conditionPath.every((segment, idx) => {
      return segment === error.conditionPath![idx];
    });
  });

  return error ? error.message : null;
}

/**
 * Format errors for a specific rule into user-friendly message
 */
export function formatRuleErrors(errors: ParsedFlatRuleError[]): string {
  if (errors.length === 0) return '';

  const lines: string[] = [];

  for (const error of errors) {
    if (error.conditionPath && error.conditionPath.length > 0) {
      // Build a human-readable path
      const pathParts: string[] = [];
      let i = 0;

      while (i < error.conditionPath.length) {
        const segment = error.conditionPath[i];

        if (segment === 'rule') {
          // Skip "rule" prefix, it's implicit
          i++;
          continue;
        }

        if (segment === 'composite' || segment === 'unary') {
          // Node type - skip it, focus on structure
          i++;
          continue;
        }

        if (segment === 'conditions') {
          // Look ahead for index
          if (i + 1 < error.conditionPath.length && typeof error.conditionPath[i + 1] === 'number') {
            const condIndex = error.conditionPath[i + 1] as number;
            pathParts.push(`Condition ${condIndex + 1}`);
            i += 2;
            continue;
          }
          i++;
          continue;
        }

        if (segment === 'then') {
          pathParts.push('Then branch');
          i++;
          continue;
        }

        if (segment === 'otherwise') {
          pathParts.push('Otherwise branch');
          i++;
          continue;
        }

        // Other segments (field names)
        if (typeof segment === 'string') {
          pathParts.push(segment);
        }

        i++;
      }

      const pathStr = pathParts.length > 0 ? pathParts.join(' → ') : 'Rule';
      lines.push(`• ${pathStr}: ${error.message}`);
    } else {
      lines.push(`• ${error.message}`);
    }
  }

  return lines.join('\n');
}

/**
 * Get section-level error flags
 */
export function getSectionErrors(parsed: ParsedFlatRuleError[]) {
  return {
    hasInputSchemaErrors: parsed.some(e => e.section === 'input_schema'),
    hasOutputErrors: parsed.some(e => e.section === 'output'),
    hasParameterErrors: parsed.some(e => e.section === 'parameters'),
  };
}
