import type { PolarsType } from '@/types/datasource-types';
import type { UITypeDef, UITypeDefStruct, UITypeDefs } from './output-types';

const CONTAINERS = ['List', 'Set', 'Array'];

export const NESTED_STRUCT_MESSAGE = "A struct can't contain another struct; use a List of it instead";
const LIST_ITEM_STRUCT_MESSAGE = 'A struct used as a List item can only have plain or categorical fields';
const NESTED_LIST_MESSAGE = 'Lists inside a struct can hold plain values, a categorical or a struct, not another List of those';

const customId = (t: PolarsType): string | null =>
  typeof t === 'object' && t !== null && t.type === 'Custom' ? t.type_id : null;

const containerInner = (t: PolarsType): PolarsType | null =>
  typeof t === 'object' && t !== null && CONTAINERS.includes(t.type) ? (t as { inner: PolarsType }).inner : null;

const isPlain = (t: PolarsType): boolean => {
  if (customId(t) !== null) return false;
  const inner = containerInner(t);
  return inner === null || isPlain(inner);
};

const isCategorical = (t: PolarsType, defs: UITypeDefs): boolean => {
  const id = customId(t);
  return id !== null && defs[id]?.type === 'categorical';
};

/** A struct that may be a List item: every field is plain (lists of plain included) or categorical. */
export const isFlatStruct = (typeDef: UITypeDef, defs: UITypeDefs): boolean =>
  typeDef.type === 'struct' &&
  typeDef.definition.fields.every(([, t]) => isPlain(t) || isCategorical(t, defs));

const isAllowedListItem = (inner: PolarsType, defs: UITypeDefs): boolean => {
  if (isPlain(inner) || isCategorical(inner, defs)) return true;
  const id = customId(inner);
  const def = id !== null ? defs[id] : undefined;
  return def?.type === 'struct' && isFlatStruct(def, defs);
};

/** Why `fieldType` can't be a field of a struct type, or null when decider accepts it. */
export const structFieldProblem = (fieldType: PolarsType, defs: UITypeDefs): string | null => {
  const id = customId(fieldType);
  if (id !== null) return defs[id]?.type === 'struct' ? NESTED_STRUCT_MESSAGE : null;
  const inner = containerInner(fieldType);
  if (inner === null || isAllowedListItem(inner, defs)) return null;
  const innerId = customId(inner);
  if (innerId !== null && defs[innerId]?.type === 'struct') return LIST_ITEM_STRUCT_MESSAGE;
  return NESTED_LIST_MESSAGE;
};

export const isAllowedStructField = (fieldType: PolarsType, defs: UITypeDefs): boolean =>
  structFieldProblem(fieldType, defs) === null;

/** Field name -> problem, for every field of `typeDef` decider would refuse. */
export const structFieldProblems = (typeDef: UITypeDefStruct, defs: UITypeDefs): Record<string, string> => {
  const problems: Record<string, string> = {};
  for (const [name, t] of typeDef.definition.fields) {
    const problem = structFieldProblem(t, defs);
    if (problem) problems[name] = problem;
  }
  return problems;
};

/**
 * Whether a custom type may be offered in a struct field's type picker.
 * `depth` is 0 for the field itself and 1 for a List/Set/Array item.
 */
export const isOfferedInStructField = (typeDef: UITypeDef, depth: number, defs: UITypeDefs): boolean => {
  if (typeDef.type === 'categorical') return depth <= 1;
  return depth === 1 && isFlatStruct(typeDef, defs);
};
