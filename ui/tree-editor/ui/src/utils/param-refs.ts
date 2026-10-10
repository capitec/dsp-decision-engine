/** A parameter reference as the editors write it. */
export type ParamRefValue = { param: string };

export const makeParamRef = (name: string): ParamRefValue => ({ param: name });

/** Name of a parameter reference, reading both `{param}` and the legacy `{key}`; null for anything else. */
export function paramRefName(value: unknown): string | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
  const v = value as { param?: unknown; key?: unknown };
  if (typeof v.param === 'string') return v.param;
  return typeof v.key === 'string' ? v.key : null;
}

export const isParamRef = (value: unknown): value is ParamRefValue => paramRefName(value) !== null;

/** A range bound as the editors print it: `$name` for a parameter reference, `-∞` for no bound. */
export function boundLabel(value: unknown): string {
  if (value === null || value === undefined) return '-∞';
  const name = paramRefName(value);
  return name !== null ? `$${name}` : String(value);
}

/**
 * Rebuild `value` with every parameter reference passed through `fn`, wherever it sits
 * (thresholds, between/range bounds, isin values, patterns, nested composites).
 * `fn(name, key)` returns the replacement (undefined keeps the reference); `key` is the property holding the reference
 * (undefined inside an array). Return `REMOVE` to drop a reference from an array.
 */
export const REMOVE = Symbol('remove');

export function mapParamRefs(
  value: any,
  fn: (name: string, key: string | undefined) => unknown,
  key?: string
): any {
  const name = paramRefName(value);
  if (name !== null) {
    const replacement = fn(name, key);
    return replacement === undefined ? value : replacement;
  }
  if (Array.isArray(value)) {
    return value
      .map(item => mapParamRefs(item, fn))
      .filter(item => item !== REMOVE);
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([k, v]) => [k, mapParamRefs(v, fn, k)])
    );
  }
  return value;
}

/** Names of all parameters referenced anywhere in `value`. */
export function collectParamRefs(value: any, into = new Set<string>()): Set<string> {
  mapParamRefs(value, name => {
    into.add(name);
  });
  return into;
}

/** Rename references to `oldName`. */
export const renameParamRefs = (value: any, oldName: string, newName: string) =>
  mapParamRefs(value, name => (name === oldName ? makeParamRef(newName) : undefined));

/** Replace references to `name` with a neutral value for the slot they sit in. */
export const clearParamRefs = (value: any, paramName: string) =>
  mapParamRefs(value, (name, key) => {
    if (name !== paramName) return undefined;
    if (key === undefined) return REMOVE;
    if (key === 'values' || key === 'patterns') return [];
    if (key === 'min' || key === 'max') return null;
    return 0;
  });
