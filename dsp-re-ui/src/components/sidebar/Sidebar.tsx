import { useEditor } from '@ctx/editor/EditorContext';
import React, { createContext, useContext, useReducer, ReactNode, useMemo } from 'react';
import styled from 'styled-components'
import BaseConditionEditor from './BaseConditionEditor'; // Import the new editor
import {
    Sidebar,
    SidebarContent,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarHeader,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
    SidebarRail,
  } from "@/components/ui/sidebar"

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

export function AppSidebar() {
    const editorState = useEditor();
    const selectedNode = useMemo(
        () => editorState.selectedNodeId && editorState.nodes[editorState.selectedNodeId],
        [editorState.selectedNodeId, editorState.nodes]
    )
    const renderNodeEditor = () => {
        if (!selectedNode || !("condition_type" in selectedNode)) {
            return <div>Value Node (TODO)</div>
        };

    

        switch (selectedNode.condition_type) {
            case 'base':
                return (
                    <BaseConditionEditor 
                        node={selectedNode} 
                        nodeId={editorState.selectedNodeId!}
                        editorState={editorState}
                    />
                );
            case 'table':
                return <div>Table Condition Editor (TODO)</div>;
            default:
                return <div>Unsupported condition type</div>;
        }
    }
    return (
        <Sidebar side="right">
            <SidebarHeader>
                {/* @ts-ignore */}
                Editor {selectedNode?.condition || "value" }
            </SidebarHeader>
            <SidebarContent>
            {renderNodeEditor()}
            </SidebarContent>
            <SidebarRail />
        </Sidebar>
        // <SideBarDiv $hidden={!editorState.selectedNodeId}>
        //     {renderNodeEditor()}
        // </SideBarDiv>
    )
}