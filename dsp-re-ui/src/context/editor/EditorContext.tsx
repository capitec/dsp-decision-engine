import React, { createContext, useReducer, useContext, Dispatch } from 'react';
import { 
  editorReducer, 
  initialEditorState 
} from './EditorReducer';
import { 
  EditorState, 
  EditorAction 
} from './editorTypes';

type EditorContextType = {
  state: EditorState;
  dispatch: Dispatch<EditorAction>;
};

const EditorContext = createContext<EditorState>(initialEditorState);
const EditorDispatchContext = createContext<Dispatch<EditorAction>>(()=>null);

export const EditorProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(editorReducer, initialEditorState);
  return (
    <EditorContext.Provider value={state}>
        <EditorDispatchContext.Provider value={dispatch}>
        {children}
        </EditorDispatchContext.Provider>
    </EditorContext.Provider>
  );
};

export const useEditor = () => {
    return useContext(EditorContext);
};

export const useEditorDispatch = () => {
    return useContext(EditorDispatchContext);
};