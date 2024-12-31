import { 
  EditorState, 
  EditorAction,
  DecisionTree,
  FlattenedNodes,
  FlattenedNodeValues,
  BaseConditionedNode,
  ChildTree,
} from './editorTypes';


import { getUID } from 'rete';

export const initialEditorState: EditorState = {
  projectId: "00000000-0000-0000-0000-000000000000",
  nodes: {},
  decision_tables: {},
  selectedNodeId: null,
  loading: false,
  error: null
};



function flattenDecisionTree(tree: DecisionTree): FlattenedNodes {
  const flattened: FlattenedNodes = {
    nodes: {},
    decision_tables: tree.decision_tables,
  };

  const dTableMaxValLookup: Record<string, number> = Object.keys(tree.decision_tables).reduce((acc,k) => {
    const table = tree.decision_tables[k];
    const maxValue = Math.max(...table.outputs["value"], table.default_value.values["value"])
    acc[k] = maxValue;
    return acc
  }, {} as Record<string, number>)

  const handleValues = (value: BaseConditionedNode["value"], parentName: string, subKey: string = 'value'): FlattenedNodeValues => {
    if (!value) {return {values:[]}}
    if (typeof value === 'string'){
      return {values:[value]}
    }
    if ("nodes" in value) {
      return recursiveFlatten(value, parentName);
    }
    if (value.type) {
      const key = getUID()
      flattened.nodes[key] = value
      return {values:[key]}
    }
    return {values:[]};
  }

  const recursiveFlatten = (value: ChildTree, parentName: string): FlattenedNodeValues => {
    const res: FlattenedNodeValues = {values: []};
    value.nodes.forEach((node) => {
      if (node.condition_type === "base"){
        const key = getUID()
        res.values.push(key);
        flattened.nodes[key] = {
          "condition_type": node.condition_type,
          "condition": node.condition,
          "connections": [handleValues(node.value, key)],
        }
      } else if (node.condition_type === "table"){ 
        const conditionTableKey = node.id || getUID()
        res.values.push(conditionTableKey);
        const childValues: FlattenedNodeValues[] = [] 
        for (let index = 0; index < (dTableMaxValLookup[node.condition_table] || 0); index++) {
          childValues.push(handleValues((node.values && node.values[index]) || null, conditionTableKey))
        }
        flattened.nodes[conditionTableKey] = {
          "condition_type": node.condition_type,
          "condition": node.condition_table,
          "connections": childValues,
        }

      }
    })
    if (value.default_value) {
      const key = getUID()
      const res = handleValues(value.default_value, key);
      flattened.nodes[key] = {
        "condition_type": "default",
        "condition": "Otherwise",
        "connections": [res]
      }
      res.defaultValue = key
    }
    
    return res
  }
  flattened.nodes["*root*"] = {
    "condition_type": "default",
    "condition": "Root",
    "connections": [recursiveFlatten(tree.root, "*root*")]
  }
  console.log({flattened})
  
  return flattened
}

export function editorReducer(state: EditorState, action: EditorAction): EditorState {
  switch (action.type) {
    case 'SET_PROJECT_ID':
      return {
        ...state,
        projectId: action.payload,
        nodes: {},
        decision_tables: {},
        error: null
      };
    case 'SELECT_NODE':
        return {
          ...state,
          selectedNodeId: action.payload,
          error: null
        };
    case 'FETCH_NODES_START':
      return {
        ...state,
        loading: true,
        error: null
      };
    case 'FETCH_NODES_SUCCESS':
      return {
        ...state,
        ...flattenDecisionTree(action.payload),
        loading: false,
        error: null
      };
    case 'FETCH_NODES_FAILURE':
      return {
        ...state,
        loading: false,
        error: action.payload
      };
    case 'CLEAR_PROJECT':
      return initialEditorState;

    case 'ADD_NODE':
      return {
        ...state,
        nodes: {
          ...state.nodes,
          [action.payload.parentKey]: {
            ...state.nodes[action.payload.parentKey],
            values: []
          }
          // If a parent key is specified, we might want to add logic to update the parent's values
          // TODO: im going to need better logic here
          //@ts-ignore
          [action.payload.parentKey || action.payload.node.condition]: action.payload.node
        }
      };

    case 'UPDATE_NODE':
      return {
        ...state,
        //@ts-ignore TODO: Going to have to see why ts doesnt like this
        nodes: {
          ...state.nodes,
          [action.payload.key]: {
            ...state.nodes[action.payload.key],
            ...action.payload.node
          }
        }
      };

    case 'DELETE_NODE':
      const { [action.payload]: deletedNode, ...remainingNodes } = state.nodes;
      return {
        ...state,
        nodes: remainingNodes
      };

    case 'ADD_DECISION_TABLE':
      return {
        ...state,
        decision_tables: {
          ...state.decision_tables,
          [action.payload.key]: action.payload.table
        }
      };

    case 'UPDATE_DECISION_TABLE':
      return {
        ...state,
        decision_tables: {
          ...state.decision_tables,
          [action.payload.key]: {
            ...state.decision_tables[action.payload.key],
            ...action.payload.table
          }
        }
      };

    case 'DELETE_DECISION_TABLE':
      const { [action.payload]: deletedTable, ...remainingTables } = state.decision_tables;
      return {
        ...state,
        decision_tables: remainingTables
      };
    default:
      return state;
  }
}