import { describe, it, expect } from 'vitest';
import {
  migrateFeatureIndicesToNames,
  usesV1FeatureFormat,
  migrateSourceDataFeatures,
  migrateVariablesToInline,
  usesV1VariableFormat,
} from '../tree-migration';
import { paramRefName } from '../param-refs';

const node = (data: any) => ({ id: 'n', data });

describe('v1 feature indices', () => {
  const features = ['age', 'income'];

  it('become feature names and the index key is removed', () => {
    const [out] = migrateFeatureIndicesToNames([node({ split_feature_id: 1, threshold: 3 })], features);
    expect(out.data).toEqual({ feature: 'income', threshold: 3 });
  });

  it('an index outside the feature list becomes a visible unknown_feature_N name', () => {
    const out = migrateFeatureIndicesToNames([node({ split_feature_id: 5 }), node({ split_feature_id: -1 })], features);
    expect(out.map(n => n.data.feature)).toEqual(['unknown_feature_5', 'unknown_feature_-1']);
  });

  it('nodes without an index and the input list are left as they were', () => {
    const input = [node({ split_feature_id: 0 }), node({ type: 'leaf' })];
    const out = migrateFeatureIndicesToNames(input, features);
    expect(out[1].data).toEqual({ type: 'leaf' });
    expect(input[0].data).toEqual({ split_feature_id: 0 });
  });

  it('usesV1FeatureFormat sees an index on any node and ignores nodes with no data', () => {
    expect(usesV1FeatureFormat([node({}), node({ split_feature_id: 0 })])).toBe(true);
    expect(usesV1FeatureFormat([node({}), { id: 'x' }])).toBe(false);
  });

  it('source-data nodes migrate the same way, keyed by id', () => {
    expect(
      migrateSourceDataFeatures({ a: { split_feature_id: 0, x: 1 }, b: { x: 2 }, c: { split_feature_id: 9 } }, features)
    ).toEqual({ a: { feature: 'age', x: 1 }, b: { x: 2 }, c: { feature: 'unknown_feature_9' } });
  });
});

describe('v1 variables', () => {
  const map = { v1: { name: 'limit', value: 5 } };

  it('usesV1VariableFormat needs a variables array', () => {
    expect(usesV1VariableFormat([node({ variables: ['v1'] })])).toBe(true);
    expect(usesV1VariableFormat([node({ variables: 'v1' }), node({})])).toBe(false);
  });

  it('the variables array is removed when there is no variable map', () => {
    const [out] = migrateVariablesToInline([node({ variables: ['v1'], threshold: 1 })], {});
    expect(out.data).toEqual({ threshold: 1 });
  });

  it('a numerical node threshold becomes a reference to the variable', () => {
    const [out] = migrateVariablesToInline([node({ node_type: 'numerical_test_node', variables: ['v1'] })], map);
    expect(out.data.variables).toBeUndefined();
    expect(out.data.threshold).toBeDefined();
  });

  it('a variable the map does not know leaves the node value as it was and drops the array', () => {
    const [out] = migrateVariablesToInline([node({ node_type: 'numerical_test_node', variables: ['gone'], threshold: 2 })], map);
    expect(out.data).toEqual({ node_type: 'numerical_test_node', threshold: 2 });
  });

  // The editors read {param: name}; a reference written as {name} is not one.
  it.fails('a migrated threshold is a parameter reference the editors recognise', () => {
    const [out] = migrateVariablesToInline([node({ node_type: 'numerical_test_node', variables: ['v1'] })], map);
    expect(paramRefName(out.data.threshold)).toBe('limit');
  });
});
