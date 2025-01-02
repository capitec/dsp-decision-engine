import React, { useMemo } from 'react';

import {
    Sidebar,
    SidebarContent,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarHeader,
    SidebarRail,
  } from "@/components/ui/sidebar"

import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select";
import { 
  Label 
} from "@/components/ui/label";

import { EditorState, TreeNode } from '@ctx/editor/editorTypes';
import { updateNode } from '@ctx/editor/editorActions';
import { useEditor, useEditorDispatch } from '@ctx/editor/EditorContext';



export function AppSidebar() {
    const editorState = useEditor();
    const selectedNode = useMemo(
        () => editorState.selectedNodeId && editorState.nodes[editorState.selectedNodeId],
        [editorState.selectedNodeId, editorState.nodes]
    )
    const editorDispatch = useEditorDispatch();
    const handleNodeTypeChange = (value: TreeNode["node_type"]) => {
      editorDispatch(updateNode(editorState.selectedNodeId!, {
        node_type: value
      }));
    };

    const renderNumericalConfig = () => {
        return (<div>Numerical</div>)
    }
    const renderCategoricalConfig = () => {
        return (<div>Categorical</div>)
    }
    const renderValueConfig = () => {
        return (<div>Value</div>)
    }

    const renderConfig = () => {
        if (!selectedNode) {
            return (<SidebarGroup>
                <SidebarGroupLabel>Project Configuration</SidebarGroupLabel>
                <SidebarGroupContent className="space-y-4 p-4"></SidebarGroupContent>
            </SidebarGroup>)
        }
        return (<SidebarGroup>
            <SidebarGroupLabel>Condition Configuration</SidebarGroupLabel>
            <SidebarGroupContent className="space-y-4 p-4">
                {/* Condition Type Dropdown */}
                <div className="space-y-2">
                <Label>Condition Type</Label>
                <Select 
                    value={selectedNode.node_type} 
                    onValueChange={handleNodeTypeChange}
                >
                    <SelectTrigger>
                    <SelectValue placeholder="Select condition type" />
                    </SelectTrigger>
                    <SelectContent>
                    <SelectItem value="numerical_test_node">Numerical</SelectItem>
                    <SelectItem value="categorical_test_node">Categorical</SelectItem>
                    <SelectItem value="leaf">Value</SelectItem>
                    </SelectContent>
                </Select>
                {
                selectedNode.node_type == 'numerical_test_node'?
                renderNumericalConfig() : 
                selectedNode.node_type == 'categorical_test_node'?
                renderCategoricalConfig() : renderValueConfig()
                }
                </div>

            </SidebarGroupContent>
        </SidebarGroup>)
    }

    return (
        <Sidebar side="right">
            <SidebarHeader>
                Header TODO get name of variable
            </SidebarHeader>
            <SidebarContent>
            {renderConfig()}
            </SidebarContent>
            <SidebarRail />
        </Sidebar>

    )
}