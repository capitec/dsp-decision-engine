import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
 
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
/** Message from a generated-client error (`{detail}` body or Error). */
export function apiErrorMessage(err: any, fallback: string): string {
  if (typeof err?.detail === 'string') return err.detail;
  return err?.message || fallback;
}

/**
 * True when `a` and `b` hold the same JSON value; key order and undefined fields do not count.
 *
 * @example
 * sameJson({ a: 1, b: undefined }, { a: 1 }) // true
 */
export function sameJson(a: any, b: any): boolean {
  if (a === b) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object' || Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(a)) return a.length === b.length && a.every((x, i) => sameJson(x, b[i]));
  const keys = (o: any) => Object.keys(o).filter((k) => o[k] !== undefined);
  const mine = keys(a);
  return mine.length === keys(b).length && mine.every((k) => sameJson(a[k], b[k]));
}
