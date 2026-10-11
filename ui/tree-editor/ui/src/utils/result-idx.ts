/**
 * Re-point every leaf's `result_idx` after output row `removed` is deleted:
 * leaves on that row become -1 (default), leaves after it shift down by one.
 * Works on tree nodes and flat-rule nodes alike.
 */
export function shiftLeafResults<T>(value: T, removed: number): T {
  if (Array.isArray(value)) return value.map(v => shiftLeafResults(v, removed)) as T;
  if (value && typeof value === 'object') {
    const obj = value as Record<string, any>;
    const out = Object.fromEntries(
      Object.entries(obj).map(([k, v]) => [k, shiftLeafResults(v, removed)])
    );
    if (obj.type === 'leaf' && typeof obj.result_idx === 'number') {
      const idx = obj.result_idx;
      out.result_idx = idx === removed ? -1 : idx > removed ? idx - 1 : idx;
    }
    return out as T;
  }
  return value;
}
