import { 
  EditorState, 
  EditorAction,
  DecisionTree,
  FlattenedNodes,
  FlattenedNodeValues,
  BaseConditionedNode,
  TableConditionedNode,
  ChildTree,
  TOutput,
  DataFrame,
  Series,
  DefaultValueNode,
} from './editorTypes';

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
      const key = `${parentName}.${subKey}`
      flattened.nodes[key] = value
      return {values:[key]}
    }
    return {values:[]};
  }

  const recursiveFlatten = (value: ChildTree, parentName: string): FlattenedNodeValues => {
    const res: FlattenedNodeValues = {values: []};
    value.nodes.forEach((node) => {
      if (node.condition_type === "base"){
        res.values.push(node.condition);
        flattened.nodes[node.condition] = {
          "condition_type": node.condition_type,
          "condition": node.condition,
          ...handleValues(node.value, node.condition),
        }
      } else if (node.condition_type === "table"){ 
        res.values.push(node.condition_table);
        const childValues: string[] = [] 
        for (let index = 0; index < (dTableMaxValLookup[node.condition_table] || 0); index++) {
          const key = `${node.condition_table}.*eq*.${index}`
          childValues.push(key)
          flattened.nodes[key] = {
            "condition_type": node.condition_type,
            "condition": `${node.condition_table} == ${index}`,
            ...handleValues((node.values && node.values[index]) || null, key),
          }
          
        }
        // const childValues = node.values?.map((v, idx)=>{
        //   console.log({v})
        //   const child = handleValues(v, node.condition_table, `values.${idx}`)
        //   return child.values[0]
        // }) || []
        flattened.nodes[node.condition_table] = {
          "condition_type": node.condition_type,
          "condition": node.condition_table,
          "values": childValues
          //...handleValues(node.value, node.condition),
        }

      }
    })
    if (value.default_value) {
      const key = `${parentName}.*default*`
      const res = handleValues(value.default_value, key);
      flattened.nodes[key] = {
        "condition_type": "default",
        "condition": "Otherwise",
        ...res
      }
      res.defaultValue = key
    }
    
    return res
  }
  flattened.nodes["*root*"] = {
    "condition_type": "default",
    "condition": "Root",
    ...recursiveFlatten(tree.root, "*root*")
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
    default:
      return state;
  }
}