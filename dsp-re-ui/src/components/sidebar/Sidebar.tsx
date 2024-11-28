import { useEditor } from '@ctx/editor/EditorContext';
import React, { createContext, useContext, useReducer, ReactNode, useMemo } from 'react';
import styled from 'styled-components'


const SideBarDiv = styled.div<{ $hidden: boolean; }>`
    height: 100vh;
    overflow: auto;
    position: fixed;
    z-index: 1;
    display: ${props => props.$hidden?"none":"block"};
    width: 400px;
    background: white;
    right: 0px;
`

export function Sidebar() {
    const editorState = useEditor();
    const selectedNode = useMemo(
        () => editorState.selectedNodeId && editorState.nodes.find(v => v.nodeId === editorState.selectedNodeId),
        [editorState.selectedNodeId, editorState.nodes]
    )
    return (
        <SideBarDiv $hidden={!editorState.selectedNodeId}>
            <h1>{selectedNode && selectedNode.name}</h1>
        </SideBarDiv>
    )
}