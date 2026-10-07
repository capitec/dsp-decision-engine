const PYTHON_KEYWORDS = new Set([
  'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue',
  'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in',
  'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield',
]);

/** Why `name` cannot be a parameter name (a Python identifier, no leading underscore, not a keyword), or null. */
export function paramNameError(name: string): string | null {
  const n = name.trim();
  if (!n) return 'Parameter name is required';
  if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(n)) {
    return `"${n}" is not a valid name: use letters, digits and underscores, starting with a letter`;
  }
  if (PYTHON_KEYWORDS.has(n)) return `"${n}" is a Python keyword`;
  return null;
}
