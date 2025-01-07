import React, { createContext, useReducer, useContext, Dispatch, useState, SetStateAction, useEffect, useCallback } from 'react';
import {
    useNodesState,
    useEdgesState,
    ReactFlowProvider,
    OnNodesChange,
    OnEdgesChange,
  } from '@xyflow/react';
import {Node, Edge, ProjectMetadata} from './types'

interface TreeOutput {
    data: string[][]
    columns: string[]
}

const defaultTreeOutput: TreeOutput = {
    data: [],
    columns: []
}

interface EditorContextState {
    features: string[];
    setFeatures: React.Dispatch<React.SetStateAction<string[]>>;
    nodes: Node[];
    setNodes: Dispatch<SetStateAction<Node[]>>;
    onNodesChange: OnNodesChange<Node>;
    edges: Edge[];
    setEdges: Dispatch<SetStateAction<Edge[]>>;
    onEdgesChange: OnEdgesChange<Edge>;
    leafOrder: string[];
    updateLeafOrder: (nodeId: string, newIndex: number) => void;
    metadata: ProjectMetadata;
    setMetadata: React.Dispatch<React.SetStateAction<ProjectMetadata>>;
    treeOutput: TreeOutput;
    setTreeOutput: React.Dispatch<React.SetStateAction<TreeOutput>>;

  }
const EditorContext = createContext<EditorContextState|undefined>(undefined);


export const EditorProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [features, setFeatures] = useState<string[]>([]);
    const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
    const [leafOrder, setLeafOrder] = useState<string[]>([]);
    const [treeOutput, setTreeOutput] = useState<EditorContextState["treeOutput"]>(defaultTreeOutput);
    const [metadata, setMetadata] = React.useState<ProjectMetadata>({
        name: "",
        description: "",
      });

    useEffect(() => {
        setLeafOrder(curr => {
            const currentLeafs: Set<string> = new Set(
                nodes
                .filter(n => n.data.node_type === "leaf")
                .map(n => n.id)
            );
    
            return [
                ...curr.filter(leaf => currentLeafs.delete(leaf)),
                ...Array.from(currentLeafs)
            ];
        });
    }, [nodes]);

    const updateLeafOrder = useCallback((nodeId: string, newIndex: number)=>{
        setLeafOrder(curr => {
            const currIndex = curr.indexOf(nodeId);
            if (currIndex === -1) { 
                console.warn(`Could not shift index of ${nodeId} to ${nodeId}`)
                return curr 
            }
            const updatedOrder = [...curr];
            updatedOrder.splice(currIndex, 1); 
            updatedOrder.splice(newIndex, 0, nodeId);
            return updatedOrder;
        })
    }, [])

    return (

    <EditorContext.Provider value={{
        features, 
        setFeatures, 

        nodes, 
        setNodes, 
        onNodesChange, 

        edges, 
        setEdges, 
        onEdgesChange,

        leafOrder,
        updateLeafOrder,

        metadata,
        setMetadata,

        treeOutput, 
        setTreeOutput,
    }}>
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
export const useLeafOrder: ()=>[EditorContextState["leafOrder"], EditorContextState["updateLeafOrder"]] = () => {
    const context = useContext(EditorContext);
    if (!context) {
        return [[], (val)=>{}]
    }
    const {leafOrder, updateLeafOrder} = context;
    return [leafOrder, updateLeafOrder];
};

export const useProjectMetadata: ()=>[EditorContextState["metadata"], EditorContextState["setMetadata"]] = () => {
    const context = useContext(EditorContext);
    if (!context) {
        return [{
            name: "",
            description: "",
          }, (val)=>{}]
    }
    const {metadata, setMetadata} = context;
    return [metadata, setMetadata];
};

export const useTreeOutput: ()=>[EditorContextState["treeOutput"], EditorContextState["setTreeOutput"]] = () => {
    const context = useContext(EditorContext);
    if (!context) {
        return [defaultTreeOutput, (val)=>{}]
    }
    const {treeOutput, setTreeOutput} = context;
    return [treeOutput, setTreeOutput,];
};