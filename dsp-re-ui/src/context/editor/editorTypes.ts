export interface DecisionTableOperation {
  predicate: number[];
  op: 'MIN' | 'MAX';
}

export interface DataFrame {
  type: 'DataFrame';
  values: Record<string, any>;
  dtypes: Record<string, string>;
}

export interface Series {
  type: 'Series';
  values: any[];
  name: string;
}

export type TOutput = DataFrame | string;

export type TCond = string; // TODO remove Series as an option doesnt really make sense to support

export interface TableCondition {
  name: string;
  table: DecisionTable;
}

export interface DecisionTable {
  operations: DecisionTableOperation[];
  operation_inputs: string[];
  outputs: {
    value: any[];
  };
  allow_multi_result: boolean;
  default_value: DataFrame;
}

export interface BaseConditionedNode {
  condition_type: 'base';
  value: TOutput | ChildTree | null;
  condition: TCond;
  priority?: number | null;
}

export interface TableConditionedNode {
  condition_type: 'table';
  id?: string;
  values: (TOutput | ChildTree)[];
  condition_table: string;
  priority: number[] | null;
}

export interface DefaultValueNode {
  condition_type: 'default';
  id?: string;
  value: TOutput | ChildTree;
  condition: TCond;
  priority: number | null;
}

export type TConditionedNode = BaseConditionedNode | TableConditionedNode;

export interface ChildTree {
  nodes: TConditionedNode[];
  default_value?: TOutput | null;
}

export interface DecisionTree {
  doc: string;
  root: ChildTree;
  decision_tables: Record<string, DecisionTable>;
}


export interface FlattenedNodeValues {
  values: string[],
  defaultValue?: string,
  priority?: number[] | null,
}

export interface FlattenedNode{
  condition_type: "base" | "table" | "default";
  condition: string,
  connections: FlattenedNodeValues[],
}

export interface FlattenedNodes {
  decision_tables: Record<string, DecisionTable>;
  nodes: Record<string, FlattenedNode | Series | DataFrame>;
}

// export interface NodeData {
//   nodeId: string;
//   name: string;
//   documentation: string;
//   dependencies: string[];
// }

export interface EditorState extends FlattenedNodes{
    projectId: string | null;
    selectedNodeId: string | null;
    loading: boolean;
    error: string | null;
}
  
export type EditorAction = 
    | { type: 'SET_PROJECT_ID'; payload: string }
    | { type: 'SELECT_NODE'; payload: string | null }
    | { type: 'FETCH_NODES_START' }
    | { type: 'FETCH_NODES_SUCCESS'; payload: DecisionTree }
    | { type: 'FETCH_NODES_FAILURE'; payload: string }
    | { type: 'CLEAR_PROJECT' }
    | { type: 'ADD_NODE'; payload: { parentKey: string; valueIdx: number; node: FlattenedNode | Series | DataFrame } }
    | { type: 'UPDATE_NODE'; payload: { key: string; node: Partial<FlattenedNode | Series | DataFrame> } }
    | { type: 'DELETE_NODE'; payload: string }
    | { type: 'ADD_DECISION_TABLE'; payload: { key: string; table: DecisionTable } }
    | { type: 'UPDATE_DECISION_TABLE'; payload: { key: string; table: Partial<DecisionTable> } }
    | { type: 'DELETE_DECISION_TABLE'; payload: string };