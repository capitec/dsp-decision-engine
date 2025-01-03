import React, { createContext, useReducer, useContext, Dispatch, useState, SetStateAction } from 'react';
import {
    useNodesState,
    useEdgesState,
    ReactFlowProvider,
    OnNodesChange,
    OnEdgesChange,
  } from '@xyflow/react';
  import {Node, Edge} from './types'

interface EditorContextState {
    features: string[];
    setFeatures: React.Dispatch<React.SetStateAction<string[]>>;
    nodes: Node[];
    setNodes: Dispatch<SetStateAction<Node[]>>;
    onNodesChange: OnNodesChange<Node>;
    edges: Edge[];
    setEdges: Dispatch<SetStateAction<Edge[]>>;
    onEdgesChange: OnEdgesChange<Edge>;
  }
const EditorContext = createContext<EditorContextState|undefined>(undefined);


export const EditorProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [features, setFeatures] = useState<string[]>([]);
    const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
    return (

    <EditorContext.Provider value={{features, setFeatures, nodes, setNodes, onNodesChange, edges, setEdges, onEdgesChange}}>
    {children}
    </EditorContext.Provider>
    );
};

export const useFeatures: ()=>[EditorContextState["features"], EditorContextState["setFeatures"]] = () => {
    const context = useContext(EditorContext);
    if (!context) {
        return [[] as string[], (val)=>{}]
    }
    const {features, setFeatures} = context;
    return [features, setFeatures];
};
export const useNodes: ()=>[EditorContextState["nodes"], EditorContextState["setNodes"], EditorContextState["onNodesChange"]] = () => {
    const context = useContext(EditorContext);
    if (!context) {
        return [[], (val)=>{}, (val)=>{}]
    }
    const {nodes, setNodes, onNodesChange} = context;
    return [nodes, setNodes, onNodesChange];
};
export const useEdges: ()=>[EditorContextState["edges"], EditorContextState["setEdges"], EditorContextState["onEdgesChange"]] = () => {
    const context = useContext(EditorContext);
    if (!context) {
        return [[], (val)=>{}, (val)=>{}]
    }
    const {edges, setEdges, onEdgesChange} = context;
    return [edges, setEdges, onEdgesChange];
};