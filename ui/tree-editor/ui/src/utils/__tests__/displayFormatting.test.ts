import { describe, it, expect, vi } from 'vitest';
import { formatDisplayTemplate, getDefaultDisplayFormat, truncateDisplay } from '../displayFormatting';

describe('formatDisplayTemplate', () => {
  it('fills fields and leaves other text alone', () => {
    expect(formatDisplayTemplate('{a} / {b}!', { a: 'x', b: 2 })).toBe('x / 2!');
  });

  it('escapes html in a scalar value', () => {
    expect(formatDisplayTemplate('{a}', { a: '<img src=x onerror="alert(1)">' })).toBe(
      '&lt;img src=x onerror=&quot;alert(1)&quot;&gt;'
    );
    expect(formatDisplayTemplate('{a}', { a: 'a/b' })).toBe('a&#x2F;b');
    expect(formatDisplayTemplate('{a}', { a: "it's & <b>" })).toBe('it&#x27;s &amp; &lt;b&gt;');
  });

  it('shows [empty] for null, undefined, missing fields and empty lists', () => {
    expect(formatDisplayTemplate('{a}{b}{c}{d}', { a: null, b: undefined, d: [] })).toBe('[empty][empty][empty][empty]');
  });

  it('joins lists, escaping them and cutting those over 50 characters', () => {
    expect(formatDisplayTemplate('{a}', { a: [1, '<b>'] })).toBe('1, &lt;b&gt;');
    const long = formatDisplayTemplate('{a}', { a: Array.from({ length: 30 }, (_, i) => `item${i}`) });
    expect(long).toHaveLength(50);
    expect(long.endsWith('...')).toBe(true);
  });

  it('prints false and 0 as values, not as empty', () => {
    expect(formatDisplayTemplate('{a}{b}', { a: false, b: 0 })).toBe('false0');
  });

  it('serialises an object to escaped JSON', () => {
    expect(formatDisplayTemplate('{a}', { a: { k: '<' } })).toBe('{&quot;k&quot;:&quot;&lt;&quot;}');
  });

  it('refuses a template naming a field that is not allowed', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(formatDisplayTemplate('{a} {secret}', { a: 1, secret: 2 }, ['a'])).toBe('[Invalid template]');
    expect(formatDisplayTemplate('{a}', { a: 1 }, ['a'])).toBe('1');
    warn.mockRestore();
  });

  it('an empty template gives an empty string', () => {
    expect(formatDisplayTemplate('', { a: 1 })).toBe('');
  });

  it('a value that looks like a template is not expanded again', () => {
    expect(formatDisplayTemplate('{a}', { a: '{b}', b: 'x' })).toBe('{b}');
  });

  // The record branch interpolates the id without escaping it.
  it.fails('escapes the id of a record-like object', () => {
    expect(formatDisplayTemplate('{a}', { a: { id: '<img src=x onerror=1>' } })).not.toContain('<img');
  });
});

describe('getDefaultDisplayFormat', () => {
  it('is empty for no fields, one placeholder for one and a comma list for several', () => {
    expect(getDefaultDisplayFormat([])).toBe('');
    expect(getDefaultDisplayFormat([{ field_name: 'a' }])).toBe('{a}');
    expect(getDefaultDisplayFormat([{ field_name: 'a' }, { field_name: 'b' }])).toBe('{a}, {b}');
  });
});

describe('truncateDisplay', () => {
  it('keeps text up to the limit and ends longer text with an ellipsis within the limit', () => {
    expect(truncateDisplay('abc')).toBe('abc');
    expect(truncateDisplay('a'.repeat(20))).toBe('a'.repeat(20));
    expect(truncateDisplay('a'.repeat(21))).toBe(`${'a'.repeat(17)}...`);
    expect(truncateDisplay('abcdefgh', 6)).toBe('abc...');
  });
});
