import { describe, it, expect } from 'vitest';
import { paramNameError } from '../param-names';

describe('paramNameError', () => {
  it.each(['a', 'rate', 'Rate2', 'max_limit', 'a_', 'x1_y2', 'ABC', 'in_', 'none', 'true'])('accepts %s', name => {
    expect(paramNameError(name)).toBeNull();
  });

  it('trims surrounding spaces before checking', () => {
    expect(paramNameError('  rate  ')).toBeNull();
  });

  it.each(['', '   ', '\t'])('requires a name: %j', name => {
    expect(paramNameError(name)).toBe('Parameter name is required');
  });

  it.each(['1a', '9', '_a', '_', '__x', 'a b', 'a-b', 'a.b', 'a$', 'é', 'naïve', 'a😀', '名前', 'a\nb'])(
    'rejects %j as not a name',
    name => {
      expect(paramNameError(name)).toContain('is not a valid name');
    }
  );

  it.each(['None', 'True', 'False', 'class', 'for', 'import', 'lambda', 'not', 'yield', 'async'])(
    'rejects the Python keyword %s',
    name => {
      expect(paramNameError(name)).toBe(`"${name}" is a Python keyword`);
    }
  );

  it('keeps the entered name in the message, trimmed', () => {
    expect(paramNameError(' 1x ')).toContain('"1x"');
  });
});
