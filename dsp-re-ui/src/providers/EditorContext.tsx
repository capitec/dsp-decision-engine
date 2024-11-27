import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import { NodeEditor, BaseSchemes } from 'rete';
import { ClassicPreset } from 'rete';

interface EditorState {
  nodes: ClassicPreset.Node[];
  //@ts-ignore
  connections: ClassicPreset.Connection[];
}

const initialState: EditorState = {
  nodes: [],
  connections: []
};

type ActionType = 
  | { type: 'ADD_NODE'; payload: ClassicPreset.Node }
  | { type: 'REMOVE_NODE'; payload: string }
  //@ts-ignore
  | { type: 'ADD_CONNECTION'; payload: ClassicPreset.Connection }
  | { type: 'REMOVE_CONNECTION'; payload: string }
  | { type: 'LOAD_GRAPH'; payload: EditorState }
  | { type: 'CLEAR_GRAPH' };

// Reducer function
function editorReducer(state: EditorState, action: ActionType): EditorState {
  switch (action.type) {
    case 'ADD_NODE':
      return { ...state, nodes: [...state.nodes, action.payload] };
    case 'REMOVE_NODE':
      return { 
        ...state, 
        nodes: state.nodes.filter(node => node.id !== action.payload),
        connections: state.connections.filter(
          conn => conn.source.node !== action.payload && conn.target.node !== action.payload
        )
      };
    case 'ADD_CONNECTION':
      return { ...state, connections: [...state.connections, action.payload] };
    case 'REMOVE_CONNECTION':
      return { 
        ...state, 
        connections: state.connections.filter(conn => conn.id !== action.payload) 
      };
    case 'LOAD_GRAPH':
      return { ...action.payload };
    case 'CLEAR_GRAPH':
      return initialState;
    default:
      return state;
  }
}

const EditorContext = createContext<{
  state: EditorState;
  dispatch: React.Dispatch<ActionType>;
  editor?: NodeEditor<BaseSchemes>;
  setEditor?: (editor: NodeEditor<BaseSchemes>) => void;
}>({
  state: initialState,
  dispatch: () => null,
});

export function EditorProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(editorReducer, initialState);
  const [editor, setEditor] = React.useState<NodeEditor<BaseSchemes>>();

  return (
    <EditorContext.Provider value={{ 
      state, 
      dispatch, 
      editor, 
      setEditor: (newEditor) => setEditor(newEditor) 
    }}>
      {children}
    </EditorContext.Provider>
  );
}

export function useEditorContext() {
  return useContext(EditorContext);
}