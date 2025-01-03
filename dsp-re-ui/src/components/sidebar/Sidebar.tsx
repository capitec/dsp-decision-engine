import React, { useMemo, useState, useCallback, useEffect } from 'react';

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

import { ReactFlow, useOnSelectionChange } from '@xyflow/react';
import { Node, Edge, TreeNode } from "../editor/types"
import { useFeatures, useNodes } from '@components/editor/EditorContext';
import { updateNodeData } from '@components/editor/util';

function useSelectedNode() {
    const [selectedNodeId, setSelectedNodeId] = useState<string|null>(null);
    const [nodes, setNodes, onNodesChange] = useNodes();

    const onChange = useCallback(({ nodes }: {nodes: Node[], edges: Edge[]}) => {
        if (nodes.length) {
            setSelectedNodeId(nodes[0].id)
        } else {
            setSelectedNodeId(null)
        }
    }, []);
    const selectedNode = useMemo(()=>{
        if (!selectedNodeId) { return null }
        return nodes.find(n=>n.id === selectedNodeId)
    }, [nodes, selectedNodeId])
     
    useOnSelectionChange({
        //@ts-ignore
        onChange, 
    });
    return selectedNode
}

export function AppSidebar() {
    const selectedNode = useSelectedNode();
    const [nodes, setNodes, onNodesChange] = useNodes();
    const [features, setFeatures] = useFeatures();

    const updateNodeType = useCallback(
        (value: TreeNode["node_type"]) => setNodes(vals => updateNodeData(selectedNode!, {node_type: value}, features, vals)),
        [selectedNode, features, setNodes],
    );

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
                    value={selectedNode.data.node_type} 
                    onValueChange={updateNodeType}
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
                selectedNode.data.node_type == 'numerical_test_node'?
                renderNumericalConfig() : 
                selectedNode.data.node_type == 'categorical_test_node'?
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