import { EditorState, EditorAction } from './editorTypes';

export const initialEditorState: EditorState = {
  projectId: "00000000-0000-0000-0000-000000000000",
  nodes: [],
  selectedNodeId: null,
  loading: false,
  error: null
};

export function editorReducer(state: EditorState, action: EditorAction): EditorState {
  switch (action.type) {
    case 'SET_PROJECT_ID':
      return {
        ...state,
        projectId: action.payload,
        nodes: [],
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
        nodes: action.payload,
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