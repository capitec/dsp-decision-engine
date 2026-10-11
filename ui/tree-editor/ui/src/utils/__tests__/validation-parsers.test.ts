import { describe, it, expect } from 'vitest';
import { parseValidationErrors, getInvalidNodeIds, formatValidationErrors } from '../validation-error-parser';
import { parseFlatRuleValidationErrors, groupErrorsByRule, hasErrorAtPath } from '../flat-rule-validation-parser';

const e = (loc: (string | number)[], msg = 'm', type = 't') => ({ loc, msg, type });
const nodes = [{ id: 'a' }, { id: 'b' }, { id: 'c' }];
const thr = ['v3', 'nodes', 0, 'data', 'unary', 'condition', '>', 'threshold'];

describe('tree validation parser', () => {
  it('groups the union branches of one field into a single error and keeps the ParamRef cause', () => {
    const parsed = parseValidationErrors(
      [
        e([...thr, 'int'], 'Input should be a valid integer'),
        e([...thr, 'float'], 'Input should be a valid number'),
        e([...thr, 'ParamRef', 'param'], "param name '1bad' must be a Python identifier"),
        e([...thr, 'ParamRef', 'name'], 'Field required'),
      ],
      nodes
    );
    expect(parsed).toHaveLength(1);
    expect(parsed[0].affectedNodeId).toBe('a');
    expect(parsed[0].fieldPath).toEqual(['condition', 'threshold']);
    expect(parsed[0].message).toContain('1bad');
  });

  it('maps node index to the saved node id and skips decikit tags', () => {
    const parsed = parseValidationErrors(
      [
        e(['v3', 'nodes', 1, 'data', 'composite'], 'a composite needs at least one condition'),
        e(['v3', 'nodes', 2, 'data', 'cases:isin', 'conditions', 0, 'values', 'list[function-before[_input_ref(), union[int,float,ParamRef]]]']),
        e(['v3', 'nodes', 2, 'data', 'cases:isin', 'conditions', 0, 'values', 'ParamRef']),
      ],
      nodes
    );
    expect(parsed).toHaveLength(2);
    expect(parsed[0].fieldPath).toEqual([]);
    expect(parsed[1].fieldPath).toEqual(['conditions', 0, 'values']);
    expect([...getInvalidNodeIds(parsed)]).toEqual(['b', 'c']);
  });

  it('highlights the node named by a build error and files unnamed ones under Other', () => {
    const parsed = parseValidationErrors(
      [
        { loc: [], msg: "tree node 'abc': result_idx 5 is not a row", type: 'build_error' },
        { loc: [], msg: 'unsupported output type', type: 'build_error' },
      ],
      nodes
    );
    expect([...getInvalidNodeIds(parsed)]).toEqual(['abc']);
    expect(formatValidationErrors(parsed)).toContain('**Other Issues:**\n• unsupported output type');
  });
});

describe('flat rule validation parser', () => {
  const base = ['prioritized_flat_rule', 'rules', 0, 'rule'];

  it('strips decikit tags so paths match the editor rule structure', () => {
    const parsed = parseFlatRuleValidationErrors([
      e([...base, 'composite', 'conditions', 0, 'unary', '<', 'threshold', 'int']),
      e([...base, 'composite', 'conditions', 0, 'unary', '<', 'threshold', 'float']),
      e([...base, 'composite', 'conditions', 0, 'unary', '<', 'threshold', 'ParamRef', 'param'], 'bad name'),
    ]);
    expect(parsed).toHaveLength(1);
    expect(parsed[0].ruleIndex).toBe(0);
    expect(parsed[0].conditionPath).toEqual(['rule', 'conditions', 0, 'threshold']);
    expect(parsed[0].message).toBe('bad name');
    expect(hasErrorAtPath(parsed, ['rule', 'conditions', 0])).toBe(true);
    expect(hasErrorAtPath(parsed, ['rule', 'conditions', 1])).toBe(false);
  });

  it('follows then/otherwise and ignores cases-condition branches of the wrong kind', () => {
    const parsed = parseFlatRuleValidationErrors([
      e([...base, 'unary', 'then', 'cases:isin', 'conditions', 0, 'when', 'StringMatchCondition', 'patterns'], 'Field required'),
      e([...base, 'unary', 'then', 'cases:isin', 'conditions', 0, 'when', 'IsInCondition', 'values', 'ParamRef'], 'bad values'),
      e([...base, 'unary', 'otherwise', 'composite', 'then', 'composite'], 'needs one'),
    ]);
    expect(parsed.map(p => p.conditionPath)).toEqual([
      ['rule', 'then', 'conditions', 0, 'when'],
      ['rule', 'otherwise', 'then'],
    ]);
    expect(groupErrorsByRule(parsed).get(0)).toHaveLength(2);
  });
});
