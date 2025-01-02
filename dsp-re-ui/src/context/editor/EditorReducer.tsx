import { 
  EditorState, 
  EditorAction,
  LeafNode
} from './editorTypes';


import { getUID } from 'rete';

export const initialEditorState: EditorState = {
  projectId: "00000000-0000-0000-0000-000000000000",
  nodes: {},
  features: [],
  selectedNodeId: null,
  loading: false,
  error: null
};

const defaultLeaf: LeafNode = {
  node_type: "leaf",
  leaf_value: 0
}



export function editorReducer(state: EditorState, action: EditorAction): EditorState {
  switch (action.type) {
    case 'SET_PROJECT_ID':
      return {
        ...initialEditorState,
        projectId: action.payload,
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
        ...action.payload,
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

    case 'UPDATE_NODE':
      const updatedNode = {
        ...state.nodes[action.payload.key],
        ...action.payload.node
      }
      const nodesUpdate = {
        [action.payload.key]: updatedNode
      }
      if (updatedNode.node_type != "leaf" ) {
        if (!updatedNode.left_child){
          const uid = getUID()
          nodesUpdate[uid] = {...defaultLeaf}
          updatedNode.left_child = uid;
        }
        if (!updatedNode.right_child){
          const uid = getUID()
          nodesUpdate[uid] = {...defaultLeaf}
          updatedNode.right_child = uid;
        }
      }
      return {
        ...state,
        //@ts-ignore TODO: Going to have to see why ts doesnt like this
        nodes: {
          ...state.nodes,
          ...nodesUpdate,
        }
      };

    default:
      return state;
  }
}