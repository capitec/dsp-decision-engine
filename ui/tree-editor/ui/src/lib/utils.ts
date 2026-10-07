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
