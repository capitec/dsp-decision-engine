import { describe, it, expect, vi, beforeEach } from 'vitest';
import flatErrors from '../../../utils/__tests__/fixtures/flat-rule-errors.json';

vi.mock('../api', () => ({ flatRuleApi: { updateProject: vi.fn() } }));
vi.mock('@/hooks/use-toast', () => ({ toast: vi.fn() }));
vi.mock('@/lib/background-token-refresh', () => ({ subscribeToStoreActivity: vi.fn() }));

import { useFlatRuleStore } from '../useFlatRuleStore';
import { flatRuleApi } from '../api';
import { toast } from '@/hooks/use-toast';
import { collectParamRefs } from '@/utils/param-refs';

const update = flatRuleApi.updateProject as unknown as ReturnType<typeof vi.fn>;
const state = () => useFlatRuleStore.getState();
const p = (param: string) => ({ param });
const leaf = (id: string, result_idx = 0) => ({ id, type: 'leaf', result_idx });

// One rule per node kind, with parameter references in every slot that holds one.
const rules = () => [
  {
    meta: { name: 'Unary', description: 'first' },
    rule: { id: 'u', type: 'unary', condition: { op: '>', feature: 'x', threshold: p('hi') }, then: leaf('u1'), otherwise: leaf('u2', -1) },
  },
  {
    meta: { name: 'Ranges' },
    rule: {
      id: 'r', type: 'cases', op: 'ranges', feature: 'x', end_logic: 'lower_inclusive', strict: false,
      conditions: [{ when: { min: null, max: p('lo') }, then: 0 }, { when: { min: p('lo'), max: 9 }, then: 1 }],
      branches: [leaf('r0'), leaf('r1'), leaf('r2', -1)],
      otherwise: 2,
    },
  },
  {
    meta: { name: 'IsIn' },
    rule: {
      id: 'i', type: 'cases', op: 'isin', feature: 'x',
      conditions: [{ when: { values: p('codes') }, then: 0 }, { when: { values: [1, 2] }, then: 1 }],
      branches: [leaf('i0'), leaf('i1'), leaf('i2', -1)],
      otherwise: 2,
    },
  },
  {
    meta: { name: 'Strings' },
    rule: {
      id: 's', type: 'cases', op: 'string_match', feature: 's', match_type: 'exact', case_sensitive: true, trim_whitespace: false,
      conditions: [{ when: { patterns: ['a', p('pat')] }, then: 0 }],
      branches: [leaf('s0'), leaf('s1', -1)],
      otherwise: 1,
    },
  },
  {
    meta: { name: 'Composite' },
    rule: {
      id: 'c', type: 'composite', op: 'or',
      conditions: [
        { type: 'unary', feature: 'x', op: '<', threshold: p('lo') },
        { type: 'composite', op: 'not', conditions: [{ type: 'unary', feature: 'x', op: 'between', min: p('lo'), max: p('hi') }] },
      ],
      then: leaf('c1'), otherwise: leaf('c2', -1),
    },
  },
];

const project = (over: any = {}): any => ({
  id: 'proj',
  name: 'P',
  project_data: {
    type: 'prioritized_flat_rule',
    input_schema: [['x', 'Float64'], ['s', 'String']],
    parameters: { hi: { type: 'Float64', default_value: 0.7 }, lo: { type: 'Float64', default_value: 0.1 } },
    output: {
      dtypes: { Action: 'String' },
      type_defs: {},
      data: [{ Action: 'Alert' }, { Action: 'Block' }],
      default: { Action: 'Allow' },
    },
    rules: rules(),
    mode: 'first_match',
    ...over,
  },
  validation_errors: [],
});

const open = (over?: any) => state().initializeFromProject(project(over));
const saved = (errors: any[] = []) => update.mockResolvedValue({ id: 'proj', validation_errors: errors });
const sentRules = (): any[] => update.mock.calls[update.mock.calls.length - 1][1].project_data.rules;

beforeEach(() => {
  update.mockReset();
  (toast as any).mockReset();
  useFlatRuleStore.setState({ saving: false });
  open();
});

describe('opening a document', () => {
  it('makes one tree per rule, named and ordered as saved, with ids on every node', () => {
    const { trees } = state();
    expect(trees.map(t => t.name)).toEqual(['Unary', 'Ranges', 'IsIn', 'Strings', 'Composite']);
    expect(trees.map(t => t.priority)).toEqual([1, 2, 3, 4, 5]);
    expect(trees[0].description).toBe('first');
    expect(new Set(trees.map(t => t.id)).size).toBe(5);
    expect(state().expandedTrees.size).toBe(5);
  });

  it('gives nodes without an id one, including nested composite conditions', () => {
    open({ rules: [{ meta: { name: 'R' }, rule: { type: 'composite', op: 'and', conditions: [{ type: 'composite', op: 'or', conditions: [] }], then: { type: 'leaf', result_idx: 0 } } }] });
    const root: any = state().trees[0].rootNode;
    expect(root.id).toEqual(expect.any(String));
    expect(root.conditions[0].id).toEqual(expect.any(String));
    expect(root.then.id).toEqual(expect.any(String));
  });

  it('keeps a cases otherwise as the saved branch index', () => {
    expect((state().trees[1].rootNode as any).otherwise).toBe(2);
  });

  it('reads the legacy strict_mode as strict, and strict wins when both are present', () => {
    open({ rules: [
      { meta: { name: 'A' }, rule: { ...rules()[1].rule, strict: undefined, strict_mode: true } },
      { meta: { name: 'B' }, rule: { ...rules()[1].rule, strict: false, strict_mode: true } },
    ] });
    const [a, b] = state().trees.map(t => t.rootNode as any);
    expect(a.strict).toBe(true);
    expect('strict_mode' in a).toBe(false);
    expect(b.strict).toBe(false);
  });

  it('opens a document with no rules on one default leaf', () => {
    open({ rules: [] });
    expect(state().trees).toHaveLength(1);
    expect(state().trees[0].rootNode).toMatchObject({ type: 'leaf', result_idx: -1 });
  });

  it('normalises dtypes given as an object and lists the output rows as options after Default', () => {
    expect(state().project.project_data.output.dtypes).toEqual([['Action', 'String']]);
    expect(state().outputOptions.map(o => [o.id, o.name])).toEqual([[-1, 'Default'], [0, 'Alert'], [1, 'Block']]);
    expect(state().availableFields).toEqual(['x', 's']);
  });

  it('puts the decikit errors on the rule they name and sets the section flags', () => {
    const errors = [...flatErrors.empty_composite, ...flatErrors.parameter_not_an_object, ...flatErrors.output_dtypes_not_a_list];
    state().initializeFromProject({ ...project(), validation_errors: errors });
    const [treeId] = [state().trees[1].id];
    expect([...state().validationErrors.keys()]).toEqual([1]);
    expect(state().ruleErrors.get(treeId)).toContain('at least one condition');
    expect(state().hasParameterErrors).toBe(true);
    expect(state().hasOutputErrors).toBe(true);
    expect(state().hasInputSchemaErrors).toBe(false);
  });

  it('ignores an error for a rule index the document does not have', () => {
    state().initializeFromProject({ ...project({ rules: rules().slice(0, 1) }), validation_errors: flatErrors.empty_composite });
    expect(state().ruleErrors.size).toBe(0);
    expect(state().validationErrors.size).toBe(0);
  });
});

describe('saving', () => {
  it('sends every rule in tree order with its name, description and the node tree as it is', async () => {
    saved();
    expect(await state().saveProject()).toBe(true);
    const sent = sentRules();
    expect(update).toHaveBeenCalledWith('proj', expect.any(Object));
    expect(sent.map(r => r.meta.name)).toEqual(['Unary', 'Ranges', 'IsIn', 'Strings', 'Composite']);
    expect(sent[0].meta).toEqual({ name: 'Unary', description: 'first', position: { x: 0, y: 0 } });
    expect(sent[1].meta).toEqual({ name: 'Ranges', position: { x: 0, y: 0 } });
    expect(sent.map(r => r.rule)).toEqual(state().trees.map(t => t.rootNode));
  });

  it('keeps every node kind: then/otherwise, ranges, isin, string patterns and nested composites', async () => {
    saved();
    await state().saveProject();
    const [unary, ranges, isin, strings, composite] = sentRules().map(r => r.rule);
    expect(unary.then.result_idx).toBe(0);
    expect(unary.otherwise.result_idx).toBe(-1);
    expect(ranges.conditions[0].when).toEqual({ min: null, max: p('lo') });
    expect(ranges.strict).toBe(false);
    expect(isin.conditions[0].when.values).toEqual(p('codes'));
    expect(strings.conditions[0].when.patterns).toEqual(['a', p('pat')]);
    expect(composite.conditions[1].conditions[0].max).toEqual(p('hi'));
  });

  it('a cases otherwise is sent as a number', async () => {
    saved();
    await state().saveProject();
    expect(sentRules().slice(1, 4).map(r => r.rule.otherwise)).toEqual([2, 2, 1]);
  });

  it('a parameter reference is sent as exactly {param}, even after node edit state changed', async () => {
    saved();
    const ranges = state().trees[1];
    state().updateNodeValue(ranges.id, 'r', 'editMode', true);
    state().updateNodeValue(ranges.id, 'r', 'isCollapsed', true);
    await state().saveProject();
    let refs = 0;
    const walk = (v: any): void => {
      if (Array.isArray(v)) return v.forEach(walk);
      if (v && typeof v === 'object') {
        if (typeof v.param === 'string') {
          refs++;
          expect(Object.keys(v)).toEqual(['param']);
        }
        Object.values(v).forEach(walk);
      }
    };
    walk(sentRules());
    expect(refs).toBeGreaterThan(5);
  });

  it('sends the rest of the document unchanged', async () => {
    saved();
    await state().saveProject();
    const data = update.mock.calls[update.mock.calls.length - 1][1].project_data;
    expect(data.parameters).toEqual(project().project_data.parameters);
    expect(data.output.data).toEqual([{ Action: 'Alert' }, { Action: 'Block' }]);
    expect(data.output.default).toEqual({ Action: 'Allow' });
    expect(data.mode).toBe('first_match');
    expect(data.input_schema).toEqual([['x', 'Float64'], ['s', 'String']]);
  });

  it('a save with validation errors still succeeds and highlights the rules', async () => {
    saved(flatErrors.empty_composite);
    expect(await state().saveProject()).toBe(true);
    expect([...state().validationErrors.keys()]).toEqual([1]);
    expect((toast as any).mock.calls[0][0].title).toBe('Saved with validation issues');
  });

  it('clears the previous errors when the next save is clean', async () => {
    saved(flatErrors.empty_composite);
    await state().saveProject();
    saved([]);
    await state().saveProject();
    expect(state().ruleErrors.size).toBe(0);
    expect(state().validationErrors.size).toBe(0);
    expect((toast as any).mock.calls[(toast as any).mock.calls.length - 1][0].title).toBe('Project Saved');
  });

  it('a failed save returns false, tells the user and frees the saving flag', async () => {
    update.mockRejectedValue({ detail: 'nope' });
    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(await state().saveProject()).toBe(false);
    expect((toast as any).mock.calls[0][0]).toMatchObject({ title: 'Save Failed', description: 'nope', variant: 'destructive' });
    expect(state().saving).toBe(false);
    err.mockRestore();
  });

  it('does not start a second save while one is running', async () => {
    let finish: (v: any) => void = () => {};
    update.mockReturnValue(new Promise(r => (finish = r)));
    const first = state().saveProject();
    expect(await state().saveProject()).toBe(false);
    finish({ id: 'proj', validation_errors: [] });
    expect(await first).toBe(true);
    expect(update).toHaveBeenCalledTimes(1);
  });
});

describe('editing', () => {
  it('updateNodeValue changes a node nested in then, otherwise and branches only', () => {
    const [unary, ranges] = state().trees;
    state().updateNodeValue(unary.id, 'u2', 'result_idx', 1);
    state().updateNodeValue(ranges.id, 'r1', 'result_idx', 1);
    expect((state().trees[0].rootNode as any).otherwise.result_idx).toBe(1);
    expect((state().trees[0].rootNode as any).then.result_idx).toBe(0);
    expect((state().trees[1].rootNode as any).branches.map((b: any) => b.result_idx)).toEqual([0, 1, -1]);
  });

  it('updateNodeValue with an unknown id or tree changes nothing', () => {
    const before = JSON.stringify(state().trees);
    state().updateNodeValue(state().trees[0].id, 'ghost', 'result_idx', 5);
    state().updateNodeValue('ghost', 'u', 'result_idx', 5);
    expect(JSON.stringify(state().trees)).toBe(before);
  });

  it('isParameterInUse sees references in conditions, branches, nested composites and then/otherwise', () => {
    for (const name of ['hi', 'lo', 'codes', 'pat']) expect(state().isParameterInUse(name), name).toBe(true);
    expect(state().isParameterInUse('unused')).toBe(false);
    state().setTrees(trees => trees.map(t => ({ ...t, rootNode: { id: 'x', type: 'unary', condition: { op: '>', feature: 'x', threshold: 1 }, then: { id: 'y', type: 'unary', condition: { op: '>', feature: 'x', threshold: p('deep') } } } as any })));
    expect(state().isParameterInUse('deep')).toBe(true);
    expect(state().isParameterInUse('hi')).toBe(false);
    expect([...collectParamRefs(state().trees[0].rootNode)]).toEqual(['deep']);
  });

  it('parameters can be created, changed, renamed with their type kept, and deleted', () => {
    state().createParameter('rate', 'Float64', 0.5);
    expect(state().project.project_data.parameters!.rate).toEqual({ type: 'Float64', default_value: 0.5 });
    state().updateParameter('rate', { value: 0.6 });
    expect(state().project.project_data.parameters!.rate).toEqual({ type: 'Float64', default_value: 0.6 });
    state().updateParameter('rate', { name: 'rate2' });
    expect(Object.keys(state().project.project_data.parameters!)).toEqual(['hi', 'lo', 'rate2']);
    expect(state().project.project_data.parameters!.rate2.default_value).toBe(0.6);
    state().deleteParameter('rate2');
    expect(Object.keys(state().project.project_data.parameters!)).toEqual(['hi', 'lo']);
  });

  it('the output default is edited at index -1 and a record at its own index', () => {
    state().saveToExistingOutput(-1, { Action: 'Deny' });
    state().saveToExistingOutput(1, { Action: 'Review' });
    state().saveAsNewOutput({ Action: 'New' });
    const out = state().project.project_data.output;
    expect(out.default).toEqual({ Action: 'Deny' });
    expect(out.data).toEqual([{ Action: 'Alert' }, { Action: 'Review' }, { Action: 'New' }]);
    expect(state().outputOptions.map(o => o.name)).toEqual(['Default', 'Alert', 'Review', 'New']);
    expect(state().outputOptions[0].description).toBe('Action: Deny');
  });
});
